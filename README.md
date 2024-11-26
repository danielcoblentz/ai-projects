# ASLConnect-ASL-Recognition-with-CNN
## Abstract
This project implements convolutional neural network models to recognize hand gestures representing numbers 0-9 in American Sign Language (ASL). Two model architectures, GestureNet and GestureNetRes, are developed and trained using TensorFlow and Keras. The models are evaluated on a custom dataset of grayscale and RGB images, achieving an accuracy of up to 95%. those models are then used to classify handgestures by the user in a seperate program

## Overview
Hand gesture recognition is a critical component in human-computer interaction, particularly for enabling communication through sign language. This project focuses on building and training CNN models to accurately classify ASL hand gestures. The models are trained on a dataset of images collected and processed specifically for this task.


![Sample Hand Gestures](/images/hand_gesture.png)
<p align="center">Figure 1: ASL hand gestures</p>

## Technologies
- Python: Version 3.10
- TensorFlow: Version 2.10.0
- Keras: Integrated with TensorFlow 2.10.0
- NumPy: For numerical computations
- OpenCV: For image processing
- Matplotlib: For plotting and visualization
- Scikit-learn: For machine learning utilities
Features

## Data Collection and Preprocessing
The data collection and preprocessing stages are critical in ensuring the models train on high-quality, representative data. Here's how we approached these stages:

- Dataset Source: We use the American Sign Language Digit Dataset available on Kaggle, which includes images of hand gestures representing digits 0-9.
- Image Acquisition: Besides the predefined dataset, we developed gather_examples.py, a script that allows users to capture images of their hand gestures using a webcam. This script guides users through capturing a balanced dataset, ensuring each gesture is well-represented.
### Preprocessing Steps:

- Grayscale Conversion: To reduce computational complexity, images are converted to grayscale, reducing the input channels from three (RGB) to one.
- Normalization: Pixel values are normalized to the range [0,1] to aid in neural network performance.
- Augmentation: To make the model robust against overfitting and to improve its ability to generalize, we apply data augmentation techniques such as rotation, zoom, and width shift.








## Dataset Information
 - [American Sign Langiage Digit Dataset](https://www.kaggle.com/datasets/rayeed045/american-sign-language-digit-dataset?resource=download)
- Training size: 5,000 images
- Validation size: 1,000 images
- Test size: 4,000 images
- Total size: 3.02 MB

## Model Architecture and Performance
### Two CNN architectures were developed:

1) GestureNet: A custom lightweight CNN suitable for real-time applications.

2) GestureNetRes: A more complex architecture incorporating residual connections to enhance learning in deeper networks.

                                                                      
                                                                       

![CNN Base Architecture](/images/CNN_architexture.png "CNN base architecture")
<p align="center">Figure 2: CNN struture</p>



## Model results
### Performance results
filter 2, model 1(Gesturenet model)
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
filter 2, model 2(Gesturenetres model)
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

#### GestureNet Model Results
Achieved an overall accuracy of 95% on the test set.
Demonstrated high precision and recall, particularly effective in distinguishing between visually similar gestures like 'six' and 'nine'.

GestureNetRes Model Results
While slightly less accurate overall at 85%, this model excelled in specific categories such as 'eight' and 'one', suggesting potential areas for further fine-tuning.

## Installation and Setup
1) Clone the Repository: Begin by cloning the repository to your local machine to get all the necessary files:
```
https://github.com/danielcoblentz/ASL-Recognition-with-CNN
```

2) Install Dependencies and activate enviorment:
```
conda activate [conda_env_name]
```
3) navigate to the directorty + train model:
after navigating to hte correct folder run hte following in the terminal.
```
python train_model.py -c config/config.json -f2 -m1
```
f1 correpsonds to the  first RGB image dataset
f2 correposnds to the second Greyscale image dataset

m1 correopsnds to model 1 (gesturenet)
m2 correopsnds to model 2 (gesturenetres)

4) once the model is trained navigate to the main program `recognize.py`:
run hte following in the terminal:
```
python recogize.py
```