from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True, help="path to face cascade resides")
# ap.add_argument("-v", "--video", help="path to video")
ap.add_argument("-m", "--model", required=True, help="path to pre-trained model")
args = vars(ap.parse_args())

detector = cv2.CascadeClassifier(args["cascade"])
model = load_model(args["model"])
'''
if not args.get("video", False):
	camera = cv2.VideoCapture(0)
else:
	camera = cv2.VideoCapture(args["video"])

while True:
	(grab, frame) = camera.read()
	if args.get("video") and not grab:
		break
	frame = imutils.resize(frame, width=300)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frameClone = frame.copy()

	rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
		minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

	for (fx, fy, fw, fh) in rects:
		roi = gray[fy:fy+fh, fx:fx+fw]
		roi = cv2.resize(roi, (28, 28))
		roi = roi.astype("float") / 255.0
		roi = img_to_array(roi)
		roi = np.expand_dims(roi, axis=0)

		(notSmiling, smiling) = model.predict(roi)[0]
		label = "Smiling" if smiling > notSmiling else "Not Smiling"

		cv2.putText(frameClone, label, (fx, fy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
		cv2.rectangle(frameClone, (fx, fy), (fx+fw, fy+fh), (0, 0, 255), 2)
	cv2.imshow("Face", frameClone)

	if cv2.waitKey(1) & 0xFF == ord("q"):
		break
'''
frame = cv2.imread("2.jpeg")
cv2.imshow("frame", frame)
cv2.waitKey(0)
# frame = imutils.resize(frame, width=300)
# frame = cv2.resize(frame, (100, 100), interpolation=cv2.INTER_CUBIC)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
frameClone = frame.copy()

rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
	minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

for (fx, fy, fw, fh) in rects:
	roi = gray[fy:fy+fh, fx:fx+fw]
	roi = cv2.resize(roi, (28, 28))
	roi = roi.astype("float") / 255.0
	roi = img_to_array(roi)
	roi = np.expand_dims(roi, axis=0)

	(notSmiling, smiling) = model.predict(roi)[0]
	label = "Smiling" if smiling > notSmiling else "Not Smiling"

	cv2.putText(frameClone, label, (fx, fy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
	cv2.rectangle(frameClone, (fx, fy), (fx+fw, fy+fh), (0, 255, 0), 2)
cv2.imshow("Face", frameClone)

cv2.waitKey(0)

cv2.destroyAllWindows()
