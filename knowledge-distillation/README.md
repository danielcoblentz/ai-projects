# Knowledge Distillation

An exploration of knowledge distillation on small image and text classifiers.

Everything here is written from scratch rather than pulled from a library: `softmax(logits / T)` for the softened targets, KL divergence from the teacher to the student, the `T^2` rescaling on the soft term, alpha mixing between the soft and hard losses, and a manual training loop.
Every distillation run is paired with a control student of identical architecture, trained on the same data, epochs, batch size and seed, but on hard labels alone, so the effect of the teacher is measured rather than assumed.
Hyperparameters are chosen on a held-out validation split, and the test set is read once per model at the end.

## Results

### MNIST digit classification

`experiments/image_classification_distillation.ipynb`.
Teacher is a 3-block CNN (1,683,722 parameters), student is a 2-block CNN (206,922 parameters), 8.1x fewer.
Five seeds per student, one teacher.

| Setting | Model | Test accuracy (mean ± std) |
| --- | --- | --- |
| 54,000 training images | Teacher | 99.10% |
| 54,000 training images | Student, control | 99.08 ± 0.08% |
| 54,000 training images | Student, distilled (`T = 4`, `alpha = 0.7`) | 98.99 ± 0.05% |
| 1,000 training images | Student, control | 95.07 ± 0.22% |
| 1,000 training images | Student, distilled (`T = 8`, `alpha = 0.5`) | 95.83 ± 0.20% |

**With the full training set, distillation did not help.**
The difference was -0.09 ± 0.11 percentage points over five paired seeds (p = 0.146).
Both students land within about 0.1 points of the teacher, so the soft targets have almost nothing left to add.

**With 1,000 training images, distillation helped on every seed.**
+0.76 ± 0.30 percentage points, ahead on 5 of 5 seeds (paired t test p = 0.004).
In the temperature and alpha sweep on the validation set, every setting beat the control, and accuracy rose with temperature while alpha mattered much less.

The student ran a batch of 128 images 8.3x faster than the teacher on CPU.

### IMDB sentiment classification

`experiments/text_classification_distillation.ipynb`.
Teacher is a bidirectional LSTM (2,766,146 parameters), student is a single-layer LSTM (675,170 parameters), 4.1x fewer.
Reviews are cut to 256 tokens, 5,000 training reviews are held out for validation, three seeds per student.

| Model | Test accuracy (mean ± std) |
| --- | --- |
| Teacher | 87.48% |
| Student, control | 85.84 ± 0.50% |
| Student, distilled (`T = 4`, `alpha = 0.7`) | 87.54 ± 0.05% |

**Distillation helped on every seed.**
+1.70 ± 0.47 percentage points, ahead on 3 of 3 seeds (paired t test p = 0.024).
The distilled student matched the teacher at a quarter of the size, and its spread over seeds was a tenth of the control's.

### Advanced techniques

`introduction/03_advanced_techniques.ipynb` measures each technique against the same control on a small MNIST setting, three seeds each.

| Technique | Minus control (pp) | Paired t test p |
| --- | --- | --- |
| Standard | +0.92 ± 0.23 | 0.020 |
| Multi-teacher (three teachers) | +1.69 ± 0.22 | 0.006 |
| Progressive temperature and alpha | +0.70 ± 0.32 | 0.064 |
| Feature-based | +1.06 ± 0.31 | 0.028 |
| Self-distillation | -0.30 ± 0.19 | 0.115 |

## Layout

```
introduction/    three notebooks working up from soft targets and temperature
                 to multi-teacher, progressive, feature-based and self-distillation
experiments/     the two measured experiments described above
distillation/    the distillation loss as a tested numpy reference
tests/           unit tests for that loss
results/         plots saved by the experiment notebooks
utils/           dataset loading and exploration notebook
```

## The loss

`distillation/losses.py` is a NumPy reference for the loss the notebooks train on, and the MNIST notebook checks its TensorFlow loss against it on random logits.

```python
soft  = mean(KL(softmax(teacher_logits / T) || softmax(student_logits / T))) * T**2
hard  = mean(cross_entropy(labels, student_logits))
total = alpha * soft + (1 - alpha) * hard
```

Both terms take raw logits, as in Hinton, Vinyals and Dean (2015), and are computed in log space.

```bash
pip install -r requirements.txt
pytest
```

The pinned versions in `requirements.txt` are the ones the saved notebook outputs were produced with, on Python 3.12 and CPU.

## Limitations

- One teacher per experiment, trained once, so the reported spread covers the students only.
- The IMDB students use `T = 4` and `alpha = 0.7` fixed in advance, with three seeds.
- On MNIST the best temperature, 8, sits at the edge of the sweep.
- The advanced techniques are compared at one small scale only.

## Next steps

- CIFAR-10, a task with real headroom at full data size, left out because a meaningful teacher and seeded students take hours on a CPU.
- A validation sweep of `T` and `alpha` for IMDB, and temperatures above 8 for MNIST.
- Several independently trained teachers per experiment.
- The LLM direction scoped in [`docs/llm-distillation-plan.md`](docs/llm-distillation-plan.md): a 7B teacher into a 0.5B student on GSM8K.
