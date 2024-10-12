# Image-classification-using-VGG16

## What is a CNN?

A Convolutional Neural Network (ConvNet/CNN) is a deep learning algorithm designed to process input images by assigning learnable weights and biases to different features or objects within the image, enabling it to distinguish one from another. In image recognition, these networks are particularly effective because they systematically identify and prioritize the most informative parts of an image, making them powerful tools for tasks such as object detection, facial recognition, and automated image labeling.

![CNN Base Architecture](https://github.com/danielcoblentz/Image-classification-using-VGG16/blob/67d47fda791a1b75400864705cf67dac0a3b3e5f/CNN%20base%20architexture.png?raw=true "CNN Base Architecture")
<p align="center">Figure 1: CNN Base Architecture</p>


## Project Description

This project leverages Convolutional Neural Networks, specifically the VGG16 model, to classify images from the Hand Gesture Recognition Database. Our focus is on fine-tuning the model's accuracy by experimenting with various hyperparameters and data augmentation techniques such as rotation and scaling. We chose this dataset for its diverse and challenging collection of hand gestures under varying conditions, which tests and enhances the model's ability to perform in real-world scenarios. By optimizing these parameters, we aim to improve gesture recognition systems, making them more adaptable and robust for practical applications in technology interfaces.
## Dataset Information

- **Link**: [Hand Gesture Recognition Database](https://www.kaggle.com/datasets/gti-upm/leapgestrecog)
- **Training size**: 20,000 images
- **Validation size**: 6,000 images
- **Test size**: 14,000 images
- **Total size**: 2 GB
- **GPU**: Google Colab A100

## Models Used
![VGG16 Architecture](https://github.com/danielcoblentz/Image-classification-using-VGG16/blob/fa02d8a4d449ed91cd636a7d099a12c7e9840df9/VGG16%20architecture.png?raw=true "VGG16 Architecture")
<p align="center">Figure 2: VGG16 Architecture</p>


We selected the VGG16 model for its architectural efficiency and proven performance in image classification tasks. Consisting of 13 convolutional layers followed by 3 fully connected layers, all utilizing 3x3 kernels, VGG16 processes images in a size of 224x224x3, making it highly suitable for detailed feature extraction in RGB or greyscale images.
- **Primary Model**: VGG16
- **Training Recommendations**: Use ‘ImageNet’, a widely known visual database for object recognition software & research. Training here before testing can yield better results and reduce training time.

## Process Breakdown


### Installation
Required packages:

```
!pip install patool==1.12
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
```

### Image Data Generator

We create an object of ImageDataGenerator its purpose is to simplify the process of importing labeled data into the model. It provides a wide range of functions, such as rescaling, rotating, zooming, and flipping images.
```python
train_dataset_raw = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.3,
    subset="training",
    seed=123,
    image_size=(224, 224),
    batch_size=batch_size,
    label_mode='categorical'
)
```
### Adam Optimizer

The **Adam optimizer** is a popular optimization algorithm used in training deep learning models. It combines the benefits of two other methods, **AdaGrad** and **RMSProp**, by adjusting the learning rate for each parameter dynamically based on the gradients.

```python
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
```
### Early stopping
Early stopping is a technique used to prevent overfitting during model training. It monitors a specific metric, such as validation loss, and stops the training process when that metric stops improving for a certain number of epochs. 
```python
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
```
### Batch normalization
Batch normalization is a technique that normalizes the inputs to each layer in a neural network, which helps stabilize and accelerate the training process. By reducing internal covariate shift, it allows the model to use higher learning rates and reduces the sensitivity to weight initialization.
```python
layers.BatchNormalization()
```





## Results and Performance

- **Accuracy**: Our model achieved an accuracy of **.9907%** 
- **Loss**: The model maintained an average loss of **0.0141**

![VGG16_result_graph](https://github.com/danielcoblentz/Image-classification-using-VGG16/blob/d989ed291b29ded1e0fafb0072e1edb852bfc157/VGG16_result_graph.png)
<p align="center">Figure 3: VGG16 graph</p>

