"""The knowledge distillation loss used by the notebooks in experiments/.

This is a NumPy port of the TensorFlow code in
experiments/image_classification_distillation.ipynb and
experiments/text_classification_distillation.ipynb. The maths is the same, so
the behaviour can be unit tested without a TensorFlow install.

One quirk is carried over deliberately. Both notebooks give the teacher and the
student a softmax output layer and then feed those outputs straight into
softmax(x / T). The scores reaching this loss are therefore probabilities, not
raw logits. That is what the notebooks measured, so it is what the tests here
pin down.
"""

import numpy as np

DEFAULT_TEMPERATURE = 4.0
DEFAULT_ALPHA = 0.7
DEFAULT_EPS = 1e-10


def softmax(scores, temperature=1.0, axis=-1):
    """Softmax over `axis`, with the scores divided by `temperature` first.

    A temperature above 1 flattens the distribution and a temperature below 1
    sharpens it. The maximum is subtracted before exponentiating so that large
    scores do not overflow.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive, got %r" % (temperature,))
    scaled = np.asarray(scores, dtype=np.float64) / temperature
    shifted = scaled - np.max(scaled, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def kl_divergence(p, q, eps=DEFAULT_EPS, axis=-1):
    """Row-wise KL(p || q) = sum(p * log(p / q)).

    `eps` is added to `q` before dividing, matching the notebooks. It keeps the
    logarithm finite when the student puts zero mass on a class, at the cost of
    a bias of roughly -eps in the result.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return np.sum(p * np.log(p / (q + eps)), axis=axis)


def soft_loss(student_scores, teacher_scores, temperature=DEFAULT_TEMPERATURE,
              eps=DEFAULT_EPS):
    """Mean KL between the softened teacher and student, rescaled by T squared.

    The gradients of the soft term scale as 1 / T^2, so multiplying by T^2 keeps
    the soft and hard terms on comparable footing as the temperature changes.
    This is the correction from section 2 of Hinton, Vinyals and Dean (2015).
    """
    teacher_soft = softmax(teacher_scores, temperature)
    student_soft = softmax(student_scores, temperature)
    per_example = kl_divergence(teacher_soft, student_soft, eps=eps)
    return float(np.mean(per_example) * (temperature ** 2))


def hard_loss(y_true, student_probs, eps=DEFAULT_EPS):
    """Mean sparse categorical cross entropy against the integer labels."""
    y_true = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(student_probs, dtype=np.float64)
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
    picked = probs[np.arange(len(y_true)), y_true]
    return float(np.mean(-np.log(np.clip(picked, eps, 1.0))))


def distillation_loss(y_true, student_scores, teacher_scores,
                      temperature=DEFAULT_TEMPERATURE, alpha=DEFAULT_ALPHA,
                      eps=DEFAULT_EPS):
    """Blend the soft and hard terms.

    `alpha` is the weight on the soft term, so alpha=1 trains on the teacher
    alone and alpha=0 ignores the teacher entirely.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1], got %r" % (alpha,))
    soft = soft_loss(student_scores, teacher_scores, temperature, eps)
    hard = hard_loss(y_true, student_scores, eps)
    return alpha * soft + (1.0 - alpha) * hard
