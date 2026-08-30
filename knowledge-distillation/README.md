# Knowledge Distillation

An exploration of knowledge distillation on small image and text classifiers.

Everything here is written from scratch rather than pulled from a library: `softmax(logits / T)` for the softened targets, a hand-written KL divergence, the `T^2` rescaling on the soft term, alpha mixing between the soft and hard losses, and a manual `GradientTape` training loop.
Every distillation run is paired with a control student of identical architecture trained on hard labels alone, so the effect of the teacher can actually be measured instead of assumed.

## Results

### MNIST digit classification

`experiments/image_classification_distillation.ipynb`.
Teacher is a 3-block CNN, student is a 2-block CNN, both ending in a softmax layer.
Trained with `T = 4.0`, `alpha = 0.7`, Adam at `lr = 0.001`, batch size 128, 10 epochs.

| Model | Parameters | Test accuracy |
| --- | --- | --- |
| Teacher (large CNN) | 1,683,722 | 99.21% |
| Student, distilled | 206,922 | 98.96% |
| Student, no distillation (control) | 206,922 | 99.13% |

The student is 8.1x smaller than the teacher, an 87.7% reduction in parameters, and 3.73x faster at inference (32.79 ms/batch against 8.78 ms/batch).

**Distillation did not help here.**
The distilled student came in 0.17 percentage points *below* the control student that never saw the teacher.
That is the honest result of the run and it is worth stating plainly rather than rounding away.

It is also not a surprising one.
MNIST is close to saturated at this scale, both students land within 0.3 points of a 99.21% teacher, and the gap is small enough that it sits inside normal run-to-run variance for a single seed.
The setup can show that a 200K-parameter student retains 99.7% of the teacher's accuracy at an eighth of the size, but it cannot resolve a difference of 0.17 points between two students.
Separating those two would need several seeds per arm and a task with enough headroom for dark knowledge to matter.

### IMDB sentiment classification

`experiments/text_classification_distillation.ipynb`.
Teacher is a bidirectional LSTM, student is a single-layer LSTM.

**This notebook has no results yet.** Its outputs were cleared rather than left in place, because the ones on disk were provably wrong.

They came from an earlier synthetic-data version and were never refreshed. Teacher, distilled student and control all reported exactly 100.00%, which is not a plausible IMDB score. The stored output claimed a 62-word vocabulary while the source loads the 10,000-word set, the two summary cells disagreed about model size (2,766,146 against 1,494,082 teacher parameters, and 675,170 against 39,138 for the student), the execution counts ran out of order (25, then 61 through 68, then 31, 32, 41, 44), and the training cell ended in a `KeyboardInterrupt`.

The source cells are untouched and the distillation maths is the same as the MNIST notebook. It needs a clean top-to-bottom re-run before anything is claimed from it.

## Layout

```
introduction/    three notebooks working up from soft targets and temperature
                 to multi-teacher, progressive, feature-based and self-distillation
experiments/     the two measured experiments described above
distillation/    the distillation loss, extracted as an importable module
tests/           unit tests for that loss
results/         plots saved by the experiment notebooks
utils/           dataset loading and exploration notebook
```

## The loss

`distillation/losses.py` holds the same loss the notebooks use, lifted out so it can be tested.

```python
soft  = mean(KL(softmax(teacher / T) || softmax(student / T))) * T**2
hard  = mean(cross_entropy(labels, student))
total = alpha * soft + (1 - alpha) * hard
```

It is a NumPy port rather than the TensorFlow original, so the tests run without a TensorFlow install.
It keeps one quirk of the notebooks on purpose: both models end in a softmax layer, so what reaches the loss and gets divided by `T` is a probability vector, not a raw logit vector.
That is what was actually measured, so that is what is pinned down.

```bash
pip install numpy pytest
pytest
```

## Next steps

In order. The first item unblocks the second.

**1. Get a TensorFlow environment.** There is none on my current machine: Python 3.14 is the only interpreter and no TensorFlow wheel exists for it, which is why the text notebook could not simply be re-run. TensorFlow also dropped native Windows GPU support after 2.10, so it is either Python 3.10 with TensorFlow 2.10 to reach the GPU, or a current Python on CPU. MNIST is small enough that CPU is fine.

**2. Re-run both notebooks end to end** and commit the real outputs. Update the results above from what they actually print, including the MNIST numbers, so every figure on this page traces to a run.

**3. Then the open questions.** Multiple seeds per arm, a temperature and alpha sweep, and CIFAR-10 as the first task with real headroom. The MNIST result is that distillation lost by 0.17 points, and a saturated task is the least informative place to measure that, so CIFAR-10 is where the question actually gets answered.

## Not done yet

Honest list of what this repo does not contain.

- **A conclusive read on whether distillation helps.** Single seed, one task with real headroom. Multiple seeds per arm and a harder dataset such as CIFAR-10 would be the first thing to fix.
- **A temperature and alpha sweep.** `T = 4.0` and `alpha = 0.7` were picked and kept. The claim that higher temperatures work better was never actually tested here.
- **A clean re-run of the text experiment**, per the note above.
- **The advanced techniques as measured experiments.** Multi-teacher, progressive, feature-based and self-distillation appear in `introduction/03_advanced_techniques.ipynb` as worked examples only.
- **Anything involving large language models.** An earlier plan for this repo was edge distillation from Qwen2.5-7B and Llama-3-8B across VQA, CIFAR-10, Flickr8k captioning and HumanEval `pass@k`. None of it was built, and it is recorded here as a direction rather than as work. The classifier experiments above are the part that exists. That sprawl is now scoped down to one runnable experiment in [`docs/llm-distillation-plan.md`](docs/llm-distillation-plan.md): a 7B teacher into a 0.5B student on GSM8K, chosen because the MNIST result suggests distillation fails on saturated tasks and a reasoning task is where the teacher's distribution should actually carry something.
