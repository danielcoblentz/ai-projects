# python train_model.py --conf config/config.json --filter 2 --model 1

# trains the asl digit classifier: --filter 1 raw images, 2 grayscale images, --model 1 cnn base, 2 cnn with residual blocks, 3 mobilenetv3
import cv2
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
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam

current_version = version.parse(tf.__version__)
print("tf version: " + tf.__version__)
desired_version = version.parse("2.10.0")
print("Version higher than 2.10.0: " + str(current_version >= desired_version))

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True,
                help="path to the input configuration file")
ap.add_argument("-f", "--filter", required=True,
                help="[the chosen filter]: filter 1 - regular image; \
	filter 2 - image with filters ")
ap.add_argument("-m", "--model", required=True,
                help="model chosen")
args = vars(ap.parse_args())

conf = Conf(args["conf"])
chosen_filter = args["filter"]
print("Filter " + chosen_filter + " has been chosen")
model_id = args["model"]
print("Model " + model_id + " has been chosen")
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
    raise argparse.ArgumentTypeError(
        "%s is not a valid model id, expected 1, 2 or 3" % model_id)

# dataset folder for each --filter value, the grayscale set is made by helper_program/converter.py
os_name = platform.system()
if os_name == "Windows":
    filter_folder = ['\\no_filter_hand_gesture_dataset', '\\filtered_hand_gesture_dataset']
else:
    filter_folder = ['/no_filter_hand_gesture_dataset', '/filtered_hand_gesture_dataset']

print("[INFO] loading images...")
datasets_root = conf["datasets_root"]
imagePaths = list(paths.list_images(datasets_root + filter_folder[int(chosen_filter)-1]))
imagePaths = [file for file in imagePaths if not os.path.basename(file).startswith('.')]
data = []
labels = []

for imagePath in imagePaths:
    # the class label is the name of the image's parent folder
    label = imagePath.split(os.path.sep)[-2]

    # grayscale 64x64 ignoring aspect ratio, except mobilenetv3 on raw images keeps 3 color channels
    image = cv2.imread(imagePath)
    if int(chosen_filter) == 1 and int(model_id) == 3:
        pass
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (64, 64))
    data.append(image)
    labels.append(label)

# scale pixel intensities to [0, 1]
data = np.array(data, dtype="float") / 255.0
# add the explicit channel dimension the conv layers expect
if int(chosen_filter) == 1 and int(model_id) == 3:
    data = data.reshape((data.shape[0], data.shape[1], data.shape[2], 3))
else:
    data = data.reshape((data.shape[0], data.shape[1], data.shape[2], 1))

# one-hot encode the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# stratified 75/25 split so every digit class shows up in both sets in proportion
(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.25, stratify=labels, random_state=42)

# augmentation for a small dataset: shifts, rotations and zooms mimic a hand drifting in the capture box
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
                         width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                         horizontal_flip=True, fill_mode="nearest")

if int(model_id) == 1 or int(model_id) == 2:
    # the custom cnns take a single grayscale channel
    model = model_name.build(64, 64, 1, len(lb.classes_))

if int(model_id) == 3 and int(chosen_filter) != 1:
    base_model = MobileNetV3Small(input_shape=(64, 64, 1), include_top=False, weights=None)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    predictions = Dense(len(lb.classes_), activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=predictions)

if int(model_id) == 3 and int(chosen_filter) == 1:
    base_model = MobileNetV3Small(input_shape=(64, 64, 3), include_top=False, weights=None)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    predictions = Dense(len(lb.classes_), activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=predictions)

print(model.summary())

# decay_steps is about one epoch's worth of batches and staircase is on, so the
# learning rate drops by 4% roughly once per epoch rather than on every step
lr_schedule = ExponentialDecay(
    initial_learning_rate=conf["init_lr"],
    decay_steps=len(trainX) // conf["bs"],
    decay_rate=0.96,
    staircase=True
)

opt = Adam(learning_rate=lr_schedule)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# steps_per_epoch is left unset so keras reads it off the finite augmentation
# iterator: a hand-set len(trainX) // bs is one batch short, which left every
# other epoch with only the leftover batch before the iterator ran dry
H = model.fit(
    aug.flow(trainX, trainY, batch_size=conf["bs"]),
    validation_data=(testX, testY),
    epochs=conf["num_epochs"]
)

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=conf["bs"])
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=lb.classes_))

print("[INFO] saving model...")
model.save(str(conf["model_path"]))

# saved so the recognizers can map output indices back to gesture names
print("[INFO] serializing label encoder...")
f = open(str(conf["lb_path"]), "wb")
f.write(pickle.dumps(lb))
f.close()
