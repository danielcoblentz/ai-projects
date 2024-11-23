# Import packages
from pyimagesearch.utils import Conf
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import os
import subprocess
import re

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-C", "--Conf", required=True,
                help="path to the input configuration file")
args = vars(ap.parse_args())

# Load the configuration file
conf = Conf(args["Conf"])

# Grab the top-left and bottom-right (x,y) coordinates for the
# gesture capture area
TOP_LEFT = tuple(conf["top_left"])
BOT_RIGHT = tuple(conf["bot_right"])

MAPPINGS = conf["mappings"]

# Loop over the mappings
for (key, label) in list(MAPPINGS.items()):
    # Update the mappings dictionary to use the original value of the key
    MAPPINGS[ord(key)] = label
    del MAPPINGS[key]

# Grab the set of valid keys from the mapping dictionary
validkeys = set(MAPPINGS.keys())

# Initialize the counter dictionary used to count the number of times
# each key has been pressed
keycounter = {}

# Start the video stream thread
print("[INFO] Starting video stream thread...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Loop over frames from the video stream
while True:
    # Grab the frame from the threaded video file stream
    frame = vs.read()

    # Resize the frame and then flip it horizontally
    frame = imutils.resize(frame, width=500)
    frame = cv2.flip(frame, 1)

    # Extract the ROI from the frame, convert it to grayscale
    # and threshold the ROI to obtain a binary mask where the 
    # foreground (white) is the hand area and the background (black)
    # should be ignored
    roi = frame[TOP_LEFT[1]:BOT_RIGHT[1], TOP_LEFT[0]:BOT_RIGHT[0]]
    #roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # roi = cv2.threshold(roi,75,255,cv2.THRESH_BINARY)[1]
    # Adaptive threshold for hand detection
    #roi = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 5)

    roi = cv2.Canny(roi, 200, 200, apertureSize=3)

    # Clone the frame and draw the capture area rectangle
    clone = frame.copy()
    cv2.rectangle(clone, TOP_LEFT, BOT_RIGHT, (0, 0, 255), 2)

    # Display the frame and ROI
    cv2.imshow("Frame", clone)
    cv2.imshow("ROI", roi)
    key = cv2.waitKey(1) & 0xFF

    # If the 'q' key was pressed, break the loop
    if key == ord("q"):
        break

    # Otherwise, check to see if a key was pressed that we are interested in capturing
    elif key in validkeys:
        # Construct the path to the local subdirectory
        p = os.path.sep.join([conf["dataset_path"], MAPPINGS[key]])

        # If the label subdirectory does not already exist, create it
        if not os.path.exists(p):
            os.makedirs(p, exist_ok=True)

        # Construct the path to the output image
        p = os.path.sep.join([p, "{}.png".format(keycounter.get(key, 0))])
        keycounter[key] = keycounter.get(key, 0) + 1

        # Save the ROI to disk
        print("[INFO] Saving ROI: {}".format(p))
        cv2.imwrite(p, roi)

# Cleanup
cv2.destroyAllWindows()
vs.stop()
