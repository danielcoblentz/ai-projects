import numpy as np
import pytest

from distillation.losses import (
    distillation_loss,
    hard_loss,
    kl_divergence,
    log_softmax,
    soft_loss,
    softmax,
)

TEACHER = np.array([[3.0, 1.0, 0.2], [0.5, 2.5, 1.0]])
STUDENT = np.array([[2.0, 1.5, 0.5], [1.0, 1.8, 1.2]])
LABELS = np.array([0, 1])


def numerical_grad(f, x, h=1e-6):
    # central differences, one coordinate at a time
    grad = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        up, down = x.copy(), x.copy()
        up[idx] += h
        down[idx] -= h
        grad[idx] = (f(up) - f(down)) / (2 * h)
    return grad


class TestSoftmax:
    def test_rows_are_probability_distributions(self):
        probs = softmax(TEACHER, temperature=2.0)
        assert np.allclose(probs.sum(axis=-1), 1.0)
        assert (probs > 0).all()

    def test_temperature_one_is_plain_softmax(self):
        scores = np.array([[1.0, 2.0, 3.0]])
        expected = np.exp(scores) / np.exp(scores).sum()
        assert np.allclose(softmax(scores, temperature=1.0), expected)

    def test_log_softmax_matches_log_of_softmax(self):
        assert np.allclose(log_softmax(TEACHER, 3.0), np.log(softmax(TEACHER, 3.0)))

    def test_higher_temperature_softens_the_distribution(self):
        sharp = softmax(TEACHER, temperature=1.0)
        soft = softmax(TEACHER, temperature=5.0)
        assert (soft.max(axis=-1) < sharp.max(axis=-1)).all()
        entropy = lambda p: -np.sum(p * np.log(p), axis=-1)
        assert (entropy(soft) > entropy(sharp)).all()

    def test_temperature_below_one_sharpens_the_distribution(self):
        sharp = softmax(TEACHER, temperature=0.5)
        assert (sharp.max(axis=-1) > softmax(TEACHER, temperature=1.0).max(axis=-1)).all()

    def test_ranking_is_preserved_across_temperatures(self):
        assert np.array_equal(
            softmax(TEACHER, temperature=0.5).argsort(axis=-1),
            softmax(TEACHER, temperature=9.0).argsort(axis=-1),
        )

    def test_large_logits_do_not_overflow(self):
        probs = softmax(np.array([[900.0, 1000.0]]), temperature=1.0)
        assert np.isfinite(probs).all()
        assert probs.sum() == pytest.approx(1.0)

    def test_non_positive_temperature_is_rejected(self):
        with pytest.raises(ValueError):
            softmax(TEACHER, temperature=0.0)


class TestKlDivergence:
    def test_is_zero_for_identical_distributions(self):
        p = softmax(TEACHER, temperature=3.0)
        assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-12)

    def test_is_non_negative_and_grows_with_disagreement(self):
        teacher = softmax(TEACHER, temperature=1.0)
        nudge = np.array([[0.05, 0.0, 0.0], [0.0, 0.05, 0.0]])
        close = kl_divergence(teacher, softmax(TEACHER + nudge, temperature=1.0))
        far = kl_divergence(teacher, softmax(STUDENT, temperature=1.0))
        assert (close >= -1e-12).all()
        assert (far > close).all()

    def test_is_invariant_to_a_constant_shift_of_the_logits(self):
        teacher = softmax(TEACHER, temperature=1.0)
        shifted = softmax(TEACHER + 3.0, temperature=1.0)
        assert kl_divergence(teacher, shifted) == pytest.approx(0.0, abs=1e-12)

    def test_is_not_symmetric(self):
        p = softmax(TEACHER, temperature=1.0)
        q = softmax(STUDENT, temperature=1.0)
        assert not np.allclose(kl_divergence(p, q), kl_divergence(q, p))

    def test_zero_teacher_probabilities_contribute_nothing(self):
        p = np.array([[1.0, 0.0, 0.0]])
        q = np.array([[0.5, 0.25, 0.25]])
        assert kl_divergence(p, q) == pytest.approx(np.log(2.0))


class TestSoftLoss:
    def test_is_zero_when_student_matches_teacher(self):
        assert soft_loss(TEACHER, TEACHER, temperature=4.0) == pytest.approx(0.0, abs=1e-12)

    def test_is_kl_of_the_softened_distributions_times_t_squared(self):
        for temperature in (1.0, 2.0, 4.0, 7.5):
            kl = np.mean(
                kl_divergence(softmax(TEACHER, temperature), softmax(STUDENT, temperature))
            )
            loss = soft_loss(STUDENT, TEACHER, temperature=temperature)
            assert loss == pytest.approx(kl * temperature ** 2)

    def test_stays_finite_when_the_teacher_is_one_hot_after_softmax(self):
        # exp(-1000) underflows to 0, the old eps based version returned nan here
        teacher = np.array([[1000.0, 0.0, 0.0]])
        loss = soft_loss(STUDENT[:1], teacher, temperature=1.0)
        assert np.isfinite(loss)
        assert loss == pytest.approx(-log_softmax(STUDENT[:1])[0, 0])

    def test_gradient_is_t_times_the_probability_gap(self):
        # d/dz_s of t^2 * kl(p_t || p_s) = t * (p_s - p_t) / batch
        temperature = 3.0
        grad = numerical_grad(lambda z: soft_loss(z, TEACHER, temperature), STUDENT.copy())
        expected = temperature * (
            softmax(STUDENT, temperature) - softmax(TEACHER, temperature)
        ) / len(STUDENT)
        assert np.allclose(grad, expected, atol=1e-6)

    def test_t_squared_keeps_the_gradient_scale_fixed_at_high_temperature(self):
        # hinton et al (2015) eq 3: for zero mean logits and large t the scaled
        # gradient tends to (z_s - z_t) / (classes * batch), independent of t
        student = STUDENT - STUDENT.mean(axis=-1, keepdims=True)
        teacher = TEACHER - TEACHER.mean(axis=-1, keepdims=True)
        limit = (student - teacher) / (student.shape[1] * len(student))
        for temperature in (50.0, 200.0):
            grad = numerical_grad(lambda z: soft_loss(z, teacher, temperature), student.copy())
            assert np.allclose(grad, limit, rtol=0.05, atol=1e-6)

    def test_without_t_squared_the_gradient_shrinks_as_one_over_t_squared(self):
        unscaled = lambda z, t: soft_loss(z, TEACHER, t) / t ** 2
        g20 = numerical_grad(lambda z: unscaled(z, 20.0), STUDENT.copy())
        g40 = numerical_grad(lambda z: unscaled(z, 40.0), STUDENT.copy())
        assert np.linalg.norm(g20) / np.linalg.norm(g40) == pytest.approx(4.0, rel=0.05)


class TestHardLoss:
    def test_matches_negative_log_softmax_of_the_true_class(self):
        log_probs = log_softmax(STUDENT)
        expected = -(log_probs[0, 0] + log_probs[1, 1]) / 2
        assert hard_loss(LABELS, STUDENT) == pytest.approx(expected)

    def test_falls_as_the_student_gets_the_label_right(self):
        confident = np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
        unsure = np.array([[0.2, 0.1, 0.0], [0.1, 0.2, 0.1]])
        assert hard_loss(LABELS, confident) < hard_loss(LABELS, unsure)

    def test_ignores_the_temperature_and_uses_t_equal_one(self):
        # the label term is always evaluated on the ordinary softmax
        assert hard_loss(LABELS, STUDENT) == pytest.approx(
            distillation_loss(LABELS, STUDENT, TEACHER, temperature=9.0, alpha=0.0)
        )

    def test_extreme_logits_stay_finite(self):
        assert np.isfinite(hard_loss(np.array([0]), np.array([[-1000.0, 1000.0]])))


class TestDistillationLoss:
    def test_alpha_one_is_the_pure_soft_label_case(self):
        combined = distillation_loss(LABELS, STUDENT, TEACHER, temperature=4.0, alpha=1.0)
        assert combined == pytest.approx(soft_loss(STUDENT, TEACHER, temperature=4.0))

    def test_alpha_zero_is_the_pure_hard_label_case(self):
        combined = distillation_loss(LABELS, STUDENT, TEACHER, temperature=4.0, alpha=0.0)
        assert combined == pytest.approx(hard_loss(LABELS, STUDENT))

    def test_alpha_zero_ignores_the_teacher_entirely(self):
        other_teacher = np.array([[0.1, 0.2, 9.0], [8.0, 0.1, 0.3]])
        assert distillation_loss(LABELS, STUDENT, TEACHER, alpha=0.0) == pytest.approx(
            distillation_loss(LABELS, STUDENT, other_teacher, alpha=0.0)
        )

    def test_is_the_weighted_blend_of_the_two_terms(self):
        alpha, temperature = 0.7, 4.0
        expected = alpha * soft_loss(STUDENT, TEACHER, temperature) + (
            1 - alpha
        ) * hard_loss(LABELS, STUDENT)
        assert distillation_loss(
            LABELS, STUDENT, TEACHER, temperature=temperature, alpha=alpha
        ) == pytest.approx(expected)

    def test_alpha_outside_the_unit_interval_is_rejected(self):
        with pytest.raises(ValueError):
            distillation_loss(LABELS, STUDENT, TEACHER, alpha=1.5)
