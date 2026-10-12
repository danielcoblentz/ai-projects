# Knowledge Distillation Experiments

Two experiments testing whether a teacher's soft targets help a smaller student, each run against a control student of identical architecture trained on hard labels alone.

- `image_classification_distillation.ipynb`: MNIST, a 3-block CNN teacher and a 2-block CNN student, five seeds, at 54,000 and at 1,000 training images, with a temperature and alpha sweep on the validation set.
- `text_classification_distillation.ipynb`: IMDB, a bidirectional LSTM teacher and a single-layer LSTM student, three seeds.

Results, settings and limitations are in the [project README](../README.md) and at the end of each notebook.
Both notebooks are saved with their outputs from a full top to bottom run.
