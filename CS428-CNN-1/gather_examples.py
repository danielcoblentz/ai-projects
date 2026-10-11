# collects training images: press a mapped key to save the current capture area under that gesture's label
from pyimagesearch.utils import Conf
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True,
                help="path to the input configuration file")
args = vars(ap.parse_args())

conf = Conf(args["conf"])

# corners of the on-screen box the hand must be held in
TOP_LEFT = tuple(conf["top_left"])
BOT_RIGHT = tuple(conf["bot_right"])

MAPPINGS = conf["mappings"]

for (key, label) in list(MAPPINGS.items()):
    # key by keycode so lookups match what cv2.waitKey returns
    MAPPINGS[ord(key)] = label
    del MAPPINGS[key]

validkeys = set(MAPPINGS.keys())

# per-label counter, used as the next image filename
keycounter = {}

# the sleep gives the camera time to warm up
print("[INFO] Starting video stream thread...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    frame = vs.read()

    # flip so the preview behaves like a mirror
    frame = imutils.resize(frame, width=500)
    frame = cv2.flip(frame, 1)

    # only the canny edge map of the capture area is saved as the training image
    roi = frame[TOP_LEFT[1]:BOT_RIGHT[1], TOP_LEFT[0]:BOT_RIGHT[0]]
    #roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # roi = cv2.threshold(roi,75,255,cv2.THRESH_BINARY)[1]
    # adaptive threshold for hand detection
    #roi = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 5)

    roi = cv2.Canny(roi, 200, 200, apertureSize=3)

    clone = frame.copy()
    cv2.rectangle(clone, TOP_LEFT, BOT_RIGHT, (0, 0, 255), 2)

    cv2.imshow("Frame", clone)
    cv2.imshow("ROI", roi)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key in validkeys:
        # one subdirectory per gesture label
        p = os.path.sep.join([conf["capture_output_path"], MAPPINGS[key]])

        if not os.path.exists(p):
            os.makedirs(p, exist_ok=True)

        p = os.path.sep.join([p, "{}.png".format(keycounter.get(key, 0))])
        keycounter[key] = keycounter.get(key, 0) + 1

        print("[INFO] Saving ROI: {}".format(p))
        cv2.imwrite(p, roi)

cv2.destroyAllWindows()
vs.stop()
