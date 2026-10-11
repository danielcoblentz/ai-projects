# Knowledge Distillation Experiments

Two experiments testing whether a teacher's soft targets help a smaller student, each run against a control student of identical architecture trained on hard labels alone.

## 1. Image classification (MNIST)

**File**: `image_classification_distillation.ipynb`

- **Task**: digit classification, 10 classes
- **Teacher**: 3-block CNN, 1,683,722 parameters
- **Student**: 2-block CNN, 206,922 parameters
- **Compression**: 8.1x fewer parameters, 87.7% smaller

Config as run:

```python
temperature = 4.0
alpha = 0.7
learning_rate = 0.001
batch_size = 128
epochs = 10
```

Measured:

| Model | Test accuracy |
| --- | --- |
| Teacher | 99.21% |
| Student, distilled | 98.96% |
| Student, no distillation | 99.13% |

Distillation improvement: **-0.17 percentage points**.
Gap to teacher: 0.25 points distilled, 0.08 points for the control.
Inference: 32.79 ms/batch teacher against 8.78 ms/batch student, a 3.73x speedup.

## 2. Text classification (IMDB sentiment)

**File**: `text_classification_distillation.ipynb`

- **Task**: binary sentiment, positive or negative
- **Teacher**: bidirectional LSTM
- **Student**: single-layer LSTM

Config in the current source:

```python
temperature = 4.0
alpha = 0.7
batch_size = 64
epochs = 3
subset_size = 5000
max_len = 500
```

**No usable numbers from this notebook yet.**
The stored outputs were cleared rather than left in place, because they were known to be wrong and a notebook with no outputs is more honest than one publishing false ones.
No source cell was changed.

What was wrong with them, recorded here and in the notebook so the evidence survives the clearing:

- Teacher, distilled student and control all reported exactly 100.00% accuracy, which is not a plausible IMDB score for any of them.
- The two summary cells disagreed on model size. One reported a 2,766,146-parameter teacher and a 675,170-parameter student, the other reported 1,494,082 and 39,138 for the same two models.
- Stored output reported a 62-word vocabulary while the current source loads the 10,000-word IMDB set with `num_words=10000`.
- The distillation training loop cell ended in a `KeyboardInterrupt` traceback, so that run never finished.
- Execution counts ran out of order (25, then 61 through 68, then 31, 32, 41, 44), so the stored state was never one top-to-bottom pass.

Those outputs came from an earlier synthetic-data version of the notebook and were never refreshed.
The notebook needs a clean top-to-bottom re-run on a machine with TensorFlow before anything is claimed from it.
TensorFlow could not be installed where the clearing was done, since no wheel exists for the Python version available there.

## What the MNIST run actually shows

The teacher did not help.
The control student beat the distilled student by 0.17 points, and both landed within 0.3 points of the teacher.

Two readings are consistent with that, and this run cannot separate them.
MNIST at 99% leaves almost no headroom for the teacher's class-similarity information to add anything, and a single seed cannot resolve a 0.17 point difference in the first place.
What the run does support is the compression claim on its own terms: 8.1x fewer parameters and 3.7x faster inference for 0.25 points of accuracy, whether the student is trained with the teacher or without it.

Temperature was fixed at 4.0 for both experiments and never swept, so this repo has nothing to say about which temperature works best.

## Results location

Plots written by the notebooks:

- `../results/image_classification/`
- `../results/text_classification/`
