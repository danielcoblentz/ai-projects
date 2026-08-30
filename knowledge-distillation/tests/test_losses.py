import numpy as np
import pytest

from distillation.losses import (
    distillation_loss,
    hard_loss,
    kl_divergence,
    soft_loss,
    softmax,
)

TEACHER = np.array([[3.0, 1.0, 0.2], [0.5, 2.5, 1.0]])
STUDENT = np.array([[2.0, 1.5, 0.5], [1.0, 1.8, 1.2]])
LABELS = np.array([0, 1])


class TestSoftmax:
    def test_rows_are_probability_distributions(self):
        probs = softmax(TEACHER, temperature=2.0)
        assert np.allclose(probs.sum(axis=-1), 1.0)
        assert (probs > 0).all()

    def test_temperature_one_is_plain_softmax(self):
        scores = np.array([[1.0, 2.0, 3.0]])
        expected = np.exp(scores) / np.exp(scores).sum()
        assert np.allclose(softmax(scores, temperature=1.0), expected)

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

    def test_large_scores_do_not_overflow(self):
        probs = softmax(np.array([[900.0, 1000.0]]), temperature=1.0)
        assert np.isfinite(probs).all()
        assert probs.sum() == pytest.approx(1.0)

    def test_non_positive_temperature_is_rejected(self):
        with pytest.raises(ValueError):
            softmax(TEACHER, temperature=0.0)


class TestKlDivergence:
    def test_is_zero_for_identical_distributions(self):
        p = softmax(TEACHER, temperature=3.0)
        assert kl_divergence(p, p, eps=0.0) == pytest.approx(0.0, abs=1e-12)

    def test_is_near_zero_with_the_epsilon_the_notebooks_use(self):
        p = softmax(TEACHER, temperature=3.0)
        assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-8)

    def test_is_non_negative_and_grows_with_disagreement(self):
        teacher = softmax(TEACHER, temperature=1.0)
        nudge = np.array([[0.05, 0.0, 0.0], [0.0, 0.05, 0.0]])
        close = kl_divergence(teacher, softmax(TEACHER + nudge, temperature=1.0), eps=0.0)
        far = kl_divergence(teacher, softmax(STUDENT, temperature=1.0), eps=0.0)
        assert (close >= -1e-12).all()
        assert (far > close).all()

    def test_is_invariant_to_a_constant_shift_of_the_scores(self):
        teacher = softmax(TEACHER, temperature=1.0)
        shifted = softmax(TEACHER + 3.0, temperature=1.0)
        assert kl_divergence(teacher, shifted, eps=0.0) == pytest.approx(0.0, abs=1e-12)

    def test_is_not_symmetric(self):
        p = softmax(TEACHER, temperature=1.0)
        q = softmax(STUDENT, temperature=1.0)
        assert not np.allclose(kl_divergence(p, q, eps=0.0), kl_divergence(q, p, eps=0.0))


class TestSoftLoss:
    def test_is_zero_when_student_matches_teacher(self):
        assert soft_loss(TEACHER, TEACHER, temperature=4.0) == pytest.approx(0.0, abs=1e-6)

    def test_applies_the_temperature_squared_rescaling(self):
        for temperature in (1.0, 2.0, 4.0, 7.5):
            unscaled = np.mean(
                kl_divergence(
                    softmax(TEACHER, temperature),
                    softmax(STUDENT, temperature),
                )
            )
            loss = soft_loss(STUDENT, TEACHER, temperature=temperature)
            assert loss == pytest.approx(unscaled * temperature ** 2)

    def test_rescaling_is_the_only_difference_from_raw_kl(self):
        at_one = soft_loss(STUDENT, TEACHER, temperature=1.0)
        raw = float(
            np.mean(kl_divergence(softmax(TEACHER, 1.0), softmax(STUDENT, 1.0)))
        )
        assert at_one == pytest.approx(raw)


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


class TestHardLoss:
    def test_matches_negative_log_of_the_true_class_probability(self):
        probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3]])
        expected = -(np.log(0.7) + np.log(0.6)) / 2
        assert hard_loss(LABELS, probs) == pytest.approx(expected)

    def test_falls_as_the_student_gets_the_label_right(self):
        confident = np.array([[0.95, 0.03, 0.02], [0.02, 0.95, 0.03]])
        unsure = np.array([[0.4, 0.35, 0.25], [0.3, 0.4, 0.3]])
        assert hard_loss(LABELS, confident) < hard_loss(LABELS, unsure)

    def test_zero_probability_on_the_true_class_stays_finite(self):
        assert np.isfinite(hard_loss(np.array([0]), np.array([[0.0, 1.0]])))
