# ASLConnect-ASL-Recognition-with-CNN
## Abstract
This project implements convolutional neural network models to recognize hand gestures representing numbers 0-9 in American Sign Language (ASL). Two model architectures, GestureNet and GestureNetRes, are developed and trained using TensorFlow and Keras, alongside a MobileNetV3Small baseline. The models are evaluated on a custom dataset of grayscale and RGB images, reaching 95% accuracy on the held-out split. A trained model is then loaded by a separate program that classifies the user's hand gestures from a webcam.

## Overview
Hand gesture recognition is a critical component in human-computer interaction, particularly for enabling communication through sign language. This project focuses on building and training CNN models to accurately classify ASL hand gestures. The models are trained on a dataset of images collected and processed specifically for this task.


![Sample Hand Gestures](/images/hand_gesture.png)
<p align="center">Figure 1: ASL hand gestures</p>

## Attribution
The scaffolding for this project is not original work.
It comes from Adrian Rosebrock's PyImageSearch gesture recognition code, published as the gesture recognition material in *Raspberry Pi for Computer Vision*, where the same capture and inference loop drives a gesture passcode security system.
`CS428-CNN-1/pyimagesearch/utils/__init__.py` still carries his `__author__` tag.
What this project adds is the model architecture work, the training pipeline and the dataset.

His code, kept here essentially unchanged:

- `CS428-CNN-1/pyimagesearch/utils/conf.py` and `pyimagesearch/utils/__init__.py`, the JSON configuration loader
- `CS428-CNN-1/pyimagesearch/notifications/`, the Twilio SMS notifier from the security system, unused here
- `CS428-CNN-1/pyimagesearch/nn/gesturenet.py`, the baseline GestureNet architecture

Adapted from that same source rather than written from scratch:

- `CS428-CNN-1/gather_examples.py`, the webcam capture loop and the key-to-label mappings; the Canny edge filtering is ours
- `CS428-CNN-1/recognize.py` and `CS428-CNN-1/recognize_tflite.py`, the live capture, ROI extraction and consecutive-frame voting loop
- `CS428-CNN-1/config/config.json`, the configuration layout
- `CS428-CNN-1/pyimagesearch/nn/gesturenetres.py`, built by adding residual connections to GestureNet

Written for this project:

- the residual blocks in `gesturenetres.py`, with 1x1 convolution shortcut projections and HeNormal initialisation
- the MobileNetV3Small branch, the ExponentialDecay learning rate schedule and the model selection flag in `train_model.py`
- the filtered against unfiltered dataset ablation behind the `--filter` flag, and the dataset itself
- TensorFlow Lite export in `tflite_convert.py` and the TFLite inference path in `recognize_tflite.py`
- `figure.py` and `helper_program/`, for plotting and for parsing training logs

## Technologies
- Python 3.10
- TensorFlow 2.10.0 or newer, with the bundled Keras; `train_model.py` checks the version at start-up
- NumPy, for numerical computations
- OpenCV and imutils, for image processing and webcam capture
- Matplotlib and pandas, for plotting the training curves
- scikit-learn, for the train/test split, label binarisation and the classification report
- json-minify, required by the configuration loader

## Data Collection and Preprocessing
The data collection and preprocessing stages are critical in ensuring the models train on high-quality, representative data. Here's how we approached these stages:

- Dataset Source: We use the American Sign Language Digit Dataset available on Kaggle, which includes images of hand gestures representing digits 0-9.
- Image Acquisition: Besides the predefined dataset, `gather_examples.py` captures extra examples from a webcam. It draws a fixed capture box, runs Canny edge detection over it, and writes the result to `datasets/raw_hand_gesture_dataset/<label>/` when you press the digit key for the gesture you are holding. Run it with `python gather_examples.py --conf config/config.json`.

### Preprocessing Steps:

- Grayscale Conversion: To reduce computational complexity, images are converted to grayscale, reducing the input channels from three (RGB) to one.
- Normalization: Pixel values are normalized to the range [0,1] to aid in neural network performance.
- Augmentation: To make the model robust against overfitting and to improve its ability to generalize, we apply data augmentation techniques such as rotation, zoom, and width shift.








## Dataset Information
Source: [American Sign Language Digit Dataset](https://www.kaggle.com/datasets/rayeed045/american-sign-language-digit-dataset?resource=download)

The copy checked in under `CS428-CNN-1/datasets/` has two variants of the same 10 classes, `zero` through `nine`:

- `no_filter_hand_gesture_dataset`, the RGB images, selected with `--filter 1`
- `filtered_hand_gesture_dataset`, the grayscale images, selected with `--filter 2`

Each variant holds 500 images per class, so 5,000 images per variant.
`train_model.py` splits a variant 75/25 with stratification, giving 3,750 training images and 1,250 held out.
There is no third split: the held-out images are passed as `validation_data` during training and then scored again for the report, which is why the support column below totals 1,250.

## Model Architecture and Performance
### Three architectures are selectable with `--model`:

1) GestureNet: A lightweight CNN suitable for real-time applications.

2) GestureNetRes: The same layout with residual connections, using 1x1 convolution shortcuts so the skip path matches the channel count of each block.

3) MobileNetV3Small: A stock Keras backbone trained from scratch, with a fresh classifier head. No results for it are committed here.

                                                                      
                                                                       

![CNN Base Architecture](/images/CNN_architexture.png "CNN base architecture")
<p align="center">Figure 2: CNN struture</p>



## Model results
Both runs below used the grayscale dataset (`--filter 2`) for 75 epochs at batch size 8.
The raw terminal output they were transcribed from is kept in `CS428-CNN-1/output/filter2_model1.txt` and `CS428-CNN-1/output/filter2_model2.txt`.

### Performance results
filter 2, model 1 (GestureNet)
| folder                    | precision         | recall                    | f1-score |  support |
|---------------------------|-------------------|---------------------------|----------|----------|
| eight                     | 0.93              | 0.94                      | 0.93     | 125      |
| five                      | 1.00              | 0.98                      | 0.99     | 125      |
| four                      | 0.93              | 0.99                      | 0.96     | 125      |
| nine                      | 0.99              | 0.90                      | 0.95     | 125      |
| one                       | 0.99              | 1.00                      | 1.00     | 125      |
| seven                     | 0.94              | 0.82                      | 0.87     | 125      |
| six                       | 0.90              | 0.87                      | 0.89     | 125      |
| three                     | 1.00              | 1.00                      | 1.00     | 125      |
| two                       | 0.84              | 0.99                      | 0.91     | 125      |
| zero                      | 1.00              | 1.00                      | 1.00     | 125      |
|                           |                   |                           |          |          |
| accuracy                  |                   |                           | 0.95     | 1250     |
| macro avg                 | 0.95              | 0.95                      | 0.95     | 1250     |
| weighted avg              | 0.95              | 0.95                      | 0.95     | 1250     |



### Performance results
filter 2, model 2 (GestureNetRes)
| folder                    | precision         | recall                    | f1-score |  support |
|---------------------------|-------------------|---------------------------|----------|----------|
| eight                     | 0.97              | 1.00                      | 0.98     | 125      |
| five                      | 1.00              | 0.84                      | 0.91     | 125      |
| four                      | 0.78              | 0.81                      | 0.80     | 125      |
| nine                      | 0.97              | 0.87                      | 0.92     | 125      |
| one                       | 0.98              | 0.99                      | 0.99     | 125      |
| seven                     | 1.00              | 0.98                      | 0.99     | 125      |
| six                       | 0.53              | 1.00                      | 0.69     | 125      |
| three                     | 1.00              | 0.42                      | 0.59     | 125      |
| two                       | 0.70              | 0.65                      | 0.67     | 125      |
| zero                      | 1.00              | 0.98                      | 0.99     | 125      |
|                           |                   |                           |          |          |
| accuracy                  |                   |                           | 0.85     | 1250     |
| macro avg                 | 0.89              | 0.85                      | 0.85     | 1250     |
| weighted avg              | 0.89              | 0.85                      | 0.85     | 1250     |

#### GestureNet
95% accuracy on the held-out split, with no class scoring below 0.87 f1.
The weakest classes are 'seven' at 0.87 and 'six' at 0.89; 'zero', 'one' and 'three' are perfect or near it.

#### GestureNetRes
85% accuracy, so worse overall than the plain CNN despite the residual connections.
It beats GestureNet on 'eight' and 'seven', but collapses on 'three' (0.42 recall) and over-predicts 'six' (0.53 precision at 1.00 recall), which is where most of the loss comes from.
The run had not settled: validation accuracy was 0.9856 at epoch 74 and 0.8536 at epoch 75.

## Installation and Setup
1) Clone the repository. It carries the full dataset and the trained weights, so the checkout is around 200 MB.
```
git clone https://github.com/danielcoblentz/asl-recognition
```

2) Activate an environment with the dependencies listed under Technologies:
```
conda activate [conda_env_name]
```

3) Change into the project directory. Every script resolves its paths relative to it:
```
cd asl-recognition/CS428-CNN-1
```

4) Train a model:
```
python train_model.py --conf config/config.json --filter 2 --model 1
```
`--filter 1` selects the RGB dataset, `--filter 2` the grayscale one.
`--model 1` is GestureNet, `--model 2` is GestureNetRes and `--model 3` is MobileNetV3Small.
Training overwrites `model_path` and `lb_path` from `config/config.json`, which by default are the committed GestureNetRes weights.

5) Run the recogniser against a webcam:
```
python recognize.py --conf config/config.json
```
To run the TensorFlow Lite build instead, export it once with `python tflite_convert.py --conf config/config.json`, then run `python recognize_tflite.py --conf config/config.json`.