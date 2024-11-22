import os
import cv2
import numpy as np
import pickle
import tensorflow as tf
from imutils.video import VideoStream
from imutils.video import FPS
import time

# Load configuration
conf = {
    "quantized_model_path": "output/gesture_reco(f2-Res).tflite",
    "lb_path": "output/gesture_lb.pickle",
    "roi_size": (64, 64),  # Change as per your model input
    "confidence_threshold": 0.5,
}

# Check if the model file exists
print("[INFO] loading model...")
if not os.path.exists(conf["quantized_model_path"]):
    print(f"[ERROR] Model file not found: {conf['quantized_model_path']}")
    exit(1)

# Load TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path=str(conf["quantized_model_path"]))
interpreter.allocate_tensors()

# Get input and output details from the model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Check if the label binarizer file exists
print("[INFO] loading label binarizer...")
if not os.path.exists(conf["lb_path"]):
    print(f"[ERROR] Label binarizer file not found: {conf['lb_path']}")
    exit(1)

# Load the label binarizer
with open(conf["lb_path"], "rb") as f:
    lb = pickle.load(f)

# Initialize the video stream and FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

# Pre-load gesture icons (if any)
icons = {
    # Add pre-loaded icons here (e.g., "thumbs_up": cv2.imread("icons/thumbs_up.png"))
}

# Process video stream frame by frame
while True:
    frame = vs.read()
    frame = cv2.flip(frame, 1)
    (h, w) = frame.shape[:2]

    # Define the region of interest (ROI)
    roi = frame[50:250, 50:250]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi, conf["roi_size"])
    roi = np.expand_dims(roi, axis=-1)
    roi = np.expand_dims(roi, axis=0)
    roi = roi.astype("float32") / 255.0

    # Make predictions using the TensorFlow Lite model
    interpreter.set_tensor(input_details[0]['index'], roi)
    interpreter.invoke()
    proba = interpreter.get_tensor(output_details[0]['index'])[0]
    label = lb.classes_[np.argmax(proba)]

    # Display prediction on the screen
    canvas = np.zeros((300, 300, 3), dtype="uint8")
    if label in icons:
        canvas[65:165, 115:215] = icons[label]
    else:
        print(f"[WARNING] Icon for gesture '{label}' not found.")
        cv2.putText(canvas, label, (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Show confidence score
    confidence = np.max(proba)
    if confidence >= conf["confidence_threshold"]:
        text = f"{label}: {confidence:.2f}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Draw the ROI box
    cv2.rectangle(frame, (50, 50), (250, 250), (0, 255, 0), 2)

    # Display the frames
    cv2.imshow("Frame", frame)
    cv2.imshow("Canvas", canvas)

    # Exit the loop on pressing 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    # Update FPS counter
    fps.update()

# Stop the FPS counter and display the results
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# Cleanup resources
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
vs.stop()
