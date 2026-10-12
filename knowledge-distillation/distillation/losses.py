"""knowledge distillation loss from the notebooks, as a tested numpy reference

every function takes raw logits, the pre-softmax outputs of the teacher and the
student, which is what hinton, vinyals and dean (2015) soften with softmax(z / t),
the tensorflow and pytorch notebooks compute the same quantities and the mnist
experiment checks its loss against this module on random logits

everything is computed in log space with a log-softmax, so no epsilon is needed
and a teacher probability that underflows to 0 contributes exactly 0
"""

import numpy as np

DEFAULT_TEMPERATURE = 4.0
DEFAULT_ALPHA = 0.7


def _check_temperature(temperature):
    if temperature <= 0:
        raise ValueError("temperature must be positive, got %r" % (temperature,))


def log_softmax(logits, temperature=1.0, axis=-1):
    """log of softmax(logits / temperature), computed stably"""
    _check_temperature(temperature)
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    shifted = scaled - np.max(scaled, axis=axis, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))


def softmax(logits, temperature=1.0, axis=-1):
    """softmax over `axis` after dividing the logits by `temperature`

    a temperature above 1 flattens the distribution and below 1 sharpens it, the
    ranking of the classes never changes, distillation uses t > 1 so the small
    probabilities the teacher gives wrong classes become large enough to learn from
    """
    return np.exp(log_softmax(logits, temperature, axis))


def kl_divergence(p, q, axis=-1):
    """row-wise kl(p || q) = sum(p * log(p / q)) for probability vectors

    the direction matters: with p the teacher and q the student, the student is
    pushed to cover every class the teacher gives mass to, classes where p is 0
    contribute 0 by the usual 0 * log 0 = 0 convention instead of nan
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    safe_p = np.where(p > 0, p, 1.0)
    with np.errstate(divide="ignore"):
        terms = np.where(p > 0, p * (np.log(safe_p) - np.log(q)), 0.0)
    return np.sum(terms, axis=axis)


def soft_loss(student_logits, teacher_logits, temperature=DEFAULT_TEMPERATURE):
    """batch mean of kl(teacher_t || student_t), rescaled by t squared

    hinton, vinyals and dean (2015, section 2): the gradients produced by soft
    targets scale as 1 / t^2, so multiplying by t^2 keeps the soft and hard terms
    on a comparable footing when the temperature changes, kl differs from the
    paper's soft cross entropy only by the teacher's entropy, a constant for the
    student, so the gradients are the same
    """
    teacher_log = log_softmax(teacher_logits, temperature)
    student_log = log_softmax(student_logits, temperature)
    per_example = np.sum(np.exp(teacher_log) * (teacher_log - student_log), axis=-1)
    return float(np.mean(per_example) * temperature ** 2)


def hard_loss(y_true, student_logits):
    """mean cross entropy of the student at t=1 against integer labels"""
    y_true = np.asarray(y_true, dtype=np.int64)
    log_probs = log_softmax(student_logits)
    return float(-np.mean(log_probs[np.arange(len(y_true)), y_true]))


def distillation_loss(y_true, student_logits, teacher_logits,
                      temperature=DEFAULT_TEMPERATURE, alpha=DEFAULT_ALPHA):
    """alpha * soft + (1 - alpha) * hard, the total loss the notebooks train on

    `alpha` is the weight on matching the teacher: alpha=1 trains on the teacher
    alone, alpha=0 ignores it and is plain supervised training, the default 0.7
    follows the paper's advice to put a considerably lower weight on the hard term
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1], got %r" % (alpha,))
    soft = soft_loss(student_logits, teacher_logits, temperature)
    hard = hard_loss(y_true, student_logits)
    return alpha * soft + (1.0 - alpha) * hard
