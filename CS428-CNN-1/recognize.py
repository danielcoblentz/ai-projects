# python recognize.py --conf config/config.json

# live asl digit recognition from the webcam using the trained keras model
import cv2
from pyimagesearch.utils import Conf
from imutils.video import VideoStream
from imutils import paths
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from datetime import datetime
import numpy as np
import argparse
import imutils
import pickle
import time
import os
import re

def showInMovedWindow(winname, img, x, y):
    cv2.namedWindow(winname)
    cv2.moveWindow(winname, x, y)
    cv2.imshow(winname,img)

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True,
    help="path to the input configuration file")
args = vars(ap.parse_args())

conf = Conf(args["conf"])


# icons shown in the prediction panel, keyed by the gesture label in the filename
print("[INFO] loading icons...")
imagePaths = paths.list_images(conf["assets_path"])
icons = {}

for imagePath in imagePaths:
    filename = os.path.basename(imagePath)
    if not re.match(r'^[a-zA-Z0-9_-]+\.png$', filename):
        print(f"[INFO] Skipping file: {filename}")
        continue
    label = filename.split(".")[0]
    icon = cv2.imread(imagePath)
  
    if icon is None:
        print(f"[ERROR] Unable to load image: {imagePath}")
        continue

    # 75x100 matches the icon slot on the prediction canvas
    try:
        icon = cv2.resize(icon, (75, 100))
    except cv2.error as e:
        print(f"[ERROR] Failed to resize image: {imagePath}, error: {e}")
        continue
    icons[label] = icon

# corners of the on-screen box the hand must be held in
TOP_LEFT = tuple(conf["top_left"])
BOT_RIGHT = tuple(conf["bot_right"])

# the label binarizer maps output indices back to gesture names
print("[INFO] loading model...")
model = load_model(str(conf["model_path"]))
lb = pickle.loads(open(str(conf["lb_path"]), "rb").read())

# the sleep gives the camera time to warm up
print("[INFO] starting video stream thread...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# [label, count] for the current run of identical predictions, a gesture is
# only accepted after consec_frames in a row so a flickering prediction is ignored
currentGesture = [None, 0]

# the gesture accepted from the user, with the timestamp it was accepted at
gestures = []
enteredTS = None

while True:
    frame = vs.read()
    timestamp = datetime.now()

    # flip so the preview behaves like a mirror
    frame = imutils.resize(frame, width=500)
    frame = cv2.flip(frame, 1)

    clone = frame.copy()
    cv2.rectangle(clone, TOP_LEFT, BOT_RIGHT, (0, 0, 255), 2)

    # skip classification while a result is still on screen
    if len(gestures) < 1:
        roi = frame[TOP_LEFT[1]:BOT_RIGHT[1], TOP_LEFT[0]:BOT_RIGHT[0]]
        
        # match the training preprocessing: grayscale, 64x64, scaled to [0, 1]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        visROI = roi.copy()
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
    
        proba = model.predict(roi)[0]
        label = lb.classes_[proba.argmax()]
        print(label)
        
        if label == currentGesture[0] and label != "ignore":
            currentGesture[1] += 1

            if currentGesture[1] == conf["consec_frames"]:
                gestures.append(label)
                currentGesture = [None, 0]

        else:
            currentGesture = [label, 0]

    # side panel showing the accepted gesture's icon and the status
    canvas = np.zeros((250, 350, 3), dtype="uint8")

    # a single icon slot, one gesture is shown at a time
    for i in range(0, 1):
        x = (i * 100) + 135

        if len(gestures) > i:
            #print (i)
            canvas[65:165, x:x + 75] = icons[gestures[i]]

        else:
            # white placeholder until a gesture is accepted
            cv2.rectangle(canvas, (x, 65), (x + 75, 165),
               (255, 255, 255), -1)

    # status shown while waiting for a gesture to be accepted
    status = "Pending Prediction ..."
    color = (255, 255, 255)

    # a gesture has been accepted, so hold the result on screen
    if len(gestures) == 1:
        if enteredTS is None:
            enteredTS = timestamp

        status = "Prediction Complete"
        color = (0, 255, 0)

        # clear it after num_seconds so the next gesture can be read
        if (timestamp - enteredTS).seconds > conf["num_seconds"]:
            gestures = []
            enteredTS = None

    ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
    status = "Status: {}".format(status)
    cv2.putText(canvas, ts, (10, canvas.shape[0] - 10),
        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 1)
    if status == "Status: Prediction Complete":
        if len(gestures) == 1:
            txt = "Prediction Result: {}".format(str(gestures[0]).capitalize())
            cv2.putText(canvas, txt, (10, canvas.shape[0] - 40),
                cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 255, 255), 1)
    cv2.putText(canvas, status, (10, 25), cv2.FONT_HERSHEY_COMPLEX,
        0.6, color, 1)

    # show the roi, the annotated frame and the prediction panel
    showInMovedWindow("ROI", visROI, 601, 0)
    showInMovedWindow("Frame", clone, 100, 0)
    showInMovedWindow("Prediction", canvas, 300, 310)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
