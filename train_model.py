# USAGE
# python train_model.py --conf config/config.json --filter 2

# import the necessary packages
import cv2
from pyimagesearch.nn.gesturenetres import GestureNetRes
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pyimagesearch.utils import Conf
from imutils import paths
import numpy as np
import argparse
import pickle
import os
import platform
import tensorflow as tf
from packaging import version
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Check TensorFlow version
current_version = version.parse(tf.__version__)
print("TensorFlow version:", tf.__version__)

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True,
                help="path to the input configuration file")
ap.add_argument("-f", "--filter", required=True,
                help="[the chosen filter]: filter 1 - regular image; \
                       filter 2 - image with filters ")
ap.add_argument("-m", "--model", required=True,
                help="model chosen")
args = vars(ap.parse_args())

# Load the configuration file
conf = Conf(args["conf"])

# Load filter type and model type
chosen_filter = args["filter"]
print("Filter " + chosen_filter + " has been chosen")
model_id = args["model"]
print("Model " + model_id + " has been chosen")

# Model selection based on input
if int(model_id) == 1:
    from pyimagesearch.nn.gesturenet import GestureNet
    model_name = GestureNet
    print("CNN Base")
elif int(model_id) == 2:
    from pyimagesearch.nn.gesturenetres import GestureNetRes
    model_name = GestureNetRes
    print("CNN+ResBlock")
elif int(model_id) == 3:
    print("MobileNetV3")
else:
    raise argparse.ArgumentTypeError(f"{model_id} is an invalid model input")

# Platform recognition and setting dataset folder paths
os_name = platform.system()
filter_folder = ['no_filter_hand_gesture_dataset', 'filtered_hand_gesture_dataset']
filter_folder = [os.path.join(conf["raw_dataset_path"], folder) for folder in filter_folder]

# Load image paths
print("[INFO] loading images...")
imagePaths = list(paths.list_images(filter_folder[int(chosen_filter) - 1]))
imagePaths = [file for file in imagePaths if not os.path.basename(file).startswith('.')]
data = []
labels = []

# Process each image path
for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    if not (int(chosen_filter) == 1 and int(model_id) == 3):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (64, 64))
    data.append(image)
    labels.append(label)

# Convert the data into a NumPy array, then preprocess it by scaling all pixel intensities to [0, 1]
data = np.array(data, dtype="float") / 255.0

# Reshape data to include channel dimension
if int(chosen_filter) == 1 and int(model_id) == 3:
    data = data.reshape((data.shape[0], data.shape[1], data.shape[2], 3))
else:
    data = data.reshape((data.shape[0], data.shape[1], data.shape[2], 1))

# One-hot encode the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# Partition data into training and testing sets
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, stratify=labels, random_state=42)

# Construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
                         width_shift_range=0.2, height_shift_range=0.2,
                         shear_range=0.15, horizontal_flip=True, fill_mode="nearest")

# Initialize the model based on selected model_id
if int(model_id) == 1 or (int(model_id) == 2 and int(chosen_filter) != 1):
    model = model_name.build(64, 64, 1, len(lb.classes_))

elif int(model_id) == 3:
    input_shape = (64, 64, 3) if int(chosen_filter) == 1 else (64, 64, 1)
    base_model = MobileNetV3Small(input_shape=input_shape, include_top=False, weights=None)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    predictions = Dense(len(lb.classes_), activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model with the Adam optimizer and categorical cross-entropy loss
print(model.summary())
opt = Adam(learning_rate=conf["init_lr"])
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train the model
H = model.fit(
    aug.flow(trainX, trainY, batch_size=conf["bs"]),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // conf["bs"],
    epochs=conf["num_epochs"]
)

# Evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=conf["bs"])
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))

# Serialize the model
print("[INFO] saving model...")
model.save(str(conf["model_path"]))

# Serialize the label encoder
print("[INFO] serializing label encoder...")
with open(str(conf["lb_path"]), "wb") as f:
    f.write(pickle.dumps(lb))
