from pyimagesearch.utils import Conf  
import tensorflow as tf
import argparse
import os

# Parse command-line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to the input configuration file")
args = vars(ap.parse_args())

# Load the configuration file
conf = Conf(args["conf"])

# Load the model and convert it to TensorFlow Lite format
model = tf.keras.models.load_model(conf["model_path"])
converter = tf.lite.TFLiteConverter.from_keras_model(model)  
tflite_model = converter.convert()

# Save the converted model
tflite_model_path = conf["model_path"].replace(".h5", ".tflite")
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)
