from config import tiny_imagenet_config as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from hdf5datasetwriter import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import json
import cv2
import os

trainPaths = list(paths.list_images(config.TRAIN_IMAGES))
print("trainPaths:\n", trainPaths[:10])
trainLabels = [p.split(os.path.sep)[-3] for p in trainPaths]
print("trainLabels:\n", trainLabels[:10])
le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)

split = train_test_split(trainPaths, trainLabels, test_size=config.NUM_TEST_IMAGES, stratify=trainLabels, random_state=42)
(trainPaths, testPaths, trainLabels, testLabels) = split

M = open(config.VAL_MAPPINGS).read().strip().split("\n")
print("original M:\n", M[:10])
M = [r.split("\t")[:2] for r in M]
print("nets M:\n", M[:10])
valPaths = [os.path.sep.join([config.VAL_IMAGES, m[0]]) for m in M]
print("val M:\n", M[:10])
valLabels = le.transform([m[1] for m in M])

datasets = [
	("train", trainPaths, trainLabels, config.TRAIN_HDF5),
	("val", valPaths, valLabels, config.VAL_HDF5),
	("test", testPaths, testLabels, config.TEST_HDF5)
]

(R, G, B) = ([], [], [])

for (dType, paths, labels, outputPath) in datasets:
	print("[INFO] building {}...".format(outputPath))
	writer = HDF5DatasetWriter((len(paths), 64, 64, 3), outputPath)
	widgets = ["Extracting Features:", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
	pbar = progressbar.ProgressBar(maxval=len(paths), widgets=widgets).start()

	for (i, (path, label)) in enumerate(zip(paths, labels)):
		image = cv2.imread(path)
		if dType == "train":
			(b, g, r) = cv2.mean(image)[:3]
			R.append(r)
			G.append(g)
			B.append(b)
		writer.add([image], [label])
		pbar.update(i)
	pbar.finish()
	writer.close()

print("[INFO] serializing means...")
D = {"R":np.mean(R), "G":np.mean(G), "B":np.mean(B)}
f = open(config.DATASET_MEAN, "w")
f.write(json.dumps(D))
f.close()
