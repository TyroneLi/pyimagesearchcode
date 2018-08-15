from imutils import paths
import argparse
import imutils
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to input directory of images")
ap.add_argument("-a", "--annot", required=True, help="path to output directory of annotations")
args = vars(ap.parse_args())

imagePaths = list(paths.list_images(args["input"]))
counts = {}

for (i, imagePath) in enumerate(imagePaths):
	print("[INFO] processing image {}/{}".format(i+1, len(imagePaths)))

	try:
		image = cv2.imread(imagePath)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		# cv2.imshow("gray-ori", gray)
		# cv2.waitKey(0)
		gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
		# cv2.imshow("gray", gray)
		# cv2.waitKey(0)
		thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
		# cv2.imshow("thresh", thresh)
		# cv2.waitKey(0)
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = cnts[0] if imutils.is_cv2() else cnts[1]
		cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]

		for c in cnts:
			(x, y, w, h) = cv2.boundingRect(c)
			roi = gray[y-5:y+h+5, x-5:x+w+5]

			cv2.imshow("ROI", imutils.resize(roi, width=60))
			# cv2.waitKey(0)
			key = cv2.waitKey(0)

			if key == ord("‘"):
				print("[INFO] ignoring character")
				continue
			key = chr(key).upper()
			dirPath = os.path.sep.join([args["annot"], key])
			if not os.path.exists(dirPath):
				os.makedirs(dirPath)

			count = counts.get(key, 1)
			p = os.path.sep.join([dirPath, "{}.png".format(str(count).zfill(6))])
			cv2.imwrite(p, roi)
			counts[key] = count + 1
	except KeyboardInterrupt:
		print("[INFO] manually leaving script")
		break
	except:
		print("[INFO] skipping image...")