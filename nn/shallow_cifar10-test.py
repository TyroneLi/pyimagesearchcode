from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from shallownet import ShallowNet
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to save train model")
args = vars(ap.parse_args())

# 标签0-9代表的类别string
labelNames = ['airplane', 'automobile', 'bird', 'cat', 
	'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

print("[INFO] loading CIFAR-10 dataset")
((trainX, trainY), (testX, testY)) = cifar10.load_data()

idxs = np.random.randint(0, len(testX), size=(10,))
testX = testX[idxs]
testY = testY[idxs]

trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

print("[INFO] loading pre-trained network...")
model = load_model(args["model"])

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32).argmax(axis=1)

print("predictions\n", predictions)

for i in range(len(testY)):
	print("label:{}".format(labelNames[predictions[i]]))

trueLabel = []
for i in range(len(testY)):
	for j in range(len(testY[i])):
		if testY[i][j] != 0:
			trueLabel.append(j)

print(trueLabel)

print("ground truth testY:")
for i in range(len(trueLabel)):
	print("label:{}".format(labelNames[trueLabel[i]]))

print("TestY\n", testY)
