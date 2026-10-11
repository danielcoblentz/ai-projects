"""knowledge distillation loss from the notebooks in experiments/, ported to numpy

the tensorflow loss in experiments/image_classification_distillation.ipynb and
experiments/text_classification_distillation.ipynb, rewritten so it can be unit
tested without a tensorflow install, the imdb notebook uses eps=1e-8 where the
mnist notebook and the default here use 1e-10

one quirk is carried over on purpose: both notebooks end the teacher and the
student in a softmax layer and then apply softmax(x / t) to those outputs, so the
scores reaching this loss are probabilities, not raw logits, and once divided by t
the softened distributions are close to uniform, the tests pin down what the
notebooks actually computed
"""

import numpy as np

DEFAULT_TEMPERATURE = 4.0
DEFAULT_ALPHA = 0.7
DEFAULT_EPS = 1e-10


def softmax(scores, temperature=1.0, axis=-1):
    """softmax over `axis` after dividing the scores by `temperature`

    a temperature above 1 flattens the distribution and below 1 sharpens it, the
    ranking of the classes never changes, distillation uses t > 1 so the small
    probabilities the teacher gives wrong classes become large enough to learn from,
    the maximum is subtracted first so large scores do not overflow
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive, got %r" % (temperature,))
    scaled = np.asarray(scores, dtype=np.float64) / temperature
    shifted = scaled - np.max(scaled, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def kl_divergence(p, q, eps=DEFAULT_EPS, axis=-1):
    """row-wise kl(p || q) = sum(p * log(p / q))

    the direction matters: with p the teacher and q the student, the student is
    pushed to cover every class the teacher gives mass to

    `eps` is added to `q`, as in the notebooks, so the log stays finite when the
    student puts zero mass on a class, this biases the result by about
    -eps * sum(p / q), which is -eps times the number of classes when p equals q
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return np.sum(p * np.log(p / (q + eps)), axis=axis)


def soft_loss(student_scores, teacher_scores, temperature=DEFAULT_TEMPERATURE,
              eps=DEFAULT_EPS):
    """batch mean of kl(teacher_t || student_t), rescaled by t squared

    hinton, vinyals and dean (2015, section 2): the gradients produced by soft
    targets scale as 1 / t^2, so multiplying by t^2 keeps the soft and hard terms
    on a comparable footing when the temperature changes, kl differs from the
    paper's soft cross entropy only by the teacher's entropy, a constant, so the
    gradients are the same
    """
    teacher_soft = softmax(teacher_scores, temperature)
    student_soft = softmax(student_scores, temperature)
    per_example = kl_divergence(teacher_soft, student_soft, eps=eps)
    return float(np.mean(per_example) * (temperature ** 2))


def hard_loss(y_true, student_probs, eps=DEFAULT_EPS):
    """mean cross entropy of the student's probabilities against integer labels

    mirrors keras sparse_categorical_crossentropy on probabilities: rows are
    renormalised to sum to 1 and clipped away from 0, keras clips at 1e-7, this
    function at `eps`
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(student_probs, dtype=np.float64)
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
    picked = probs[np.arange(len(y_true)), y_true]
    return float(np.mean(-np.log(np.clip(picked, eps, 1.0))))


def distillation_loss(y_true, student_scores, teacher_scores,
                      temperature=DEFAULT_TEMPERATURE, alpha=DEFAULT_ALPHA,
                      eps=DEFAULT_EPS):
    """alpha * soft + (1 - alpha) * hard, the total loss the notebooks train on

    `alpha` is the weight on matching the teacher: alpha=1 trains on the teacher
    alone, alpha=0 ignores it and is plain supervised training, the default 0.7
    follows the paper's advice to put a considerably lower weight on the hard term
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1], got %r" % (alpha,))
    soft = soft_loss(student_scores, teacher_scores, temperature, eps)
    hard = hard_loss(y_true, student_scores, eps)
    return alpha * soft + (1.0 - alpha) * hard
