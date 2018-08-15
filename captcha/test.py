from keras.preprocessing.image import img_to_array
from keras.models import load_model
from captchahelper import preprocess
from imutils import contours
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to input dataset")
ap.add_argument("-m", "--model", required=True, help="path to output model")
args = vars(ap.parse_args())

print("[INFO] loading pre-trained network...")
model = load_model(args["model"])

imagePaths = list(paths.list_images(args["input"]))
imagePaths = np.random.choice(imagePaths, size=(5,), replace=False)

for imagePath in imagePaths:
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.copyMakeBorder(gray, 20, 20, 20, 20, cv2.BORDER_REPLICATE)

	thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]
	cnts = contours.sort_contours(cnts)[0]

	output = cv2.merge([gray] * 3)
	predictions = []

	for c in cnts:
			(x, y, w, h) = cv2.boundingRect(c)
			roi = gray[y-5:y+h+5, x-5:x+w+5]
			roi = preprocess(roi, 28, 28)
			roi = np.expand_dims(img_to_array(roi), axis=0) / 255.0
			pred = model.predict(roi).argmax(axis=1)[0] + 1
			predictions.append(str(pred))

			cv2.rectangle(output, (x-2, y-2), (x+w+4, y+h+4), (0, 255, 0), 1)
			cv2.putText(output, str(pred), (x-5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

			print("[INFO] captcha: {}".format("".join(predictions)))
			cv2.imshow("output", output)
			cv2.waitKey(0)
