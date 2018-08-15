import matplotlib
matplotlib.use("Agg")
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from miniVGG import MiniVGGNet
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from trainingmonitor import TrainingMonitor
from keras.datasets import cifar10
from keras.datasets import cifar100
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def step_decay(epoch):
	initAlpha = 0.01
	factor = 0.5
	dropEvery = 5

	alpha = initAlpha * (factor ** np.floor((1 + epoch) / dropEvery))

	return float(alpha)

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to the output loss/accuracy plot")
ap.add_argument("-m", "--model", required=True, help="path to save train model")
args = vars(ap.parse_args())

print("[INFO] process ID: {}".format(os.getpid()))

print("[INFO] loading CIFAR-10 dataset")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)
# 标签0-9代表的类别string
labelNames = ['airplane', 'automobile', 'bird', 'cat', 
	'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

print("[INFO] compiling model...")
# callbacks = [LearningRateScheduler(step_decay)]
# opt = SGD(lr=0.01, decay=0.01 / 70, momentum=0.9, nesterov=True)
opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])
print(model.summary())

figPath = os.path.sep.join([args["output"], "{}.png".format(os.getpid())])
jsonPath = os.path.sep.join([args["output"], "{}.json".format(os.getpid())])

callbacks = [TrainingMonitor(figPath, jsonPath=jsonPath)]

print("[INFO] training network Lenet-5")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=100, 
	callbacks=callbacks, verbose=1)

model.save(args["model"])

print("[INFO] evaluating Lenet-5..")
preds = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1), 
	target_names=labelNames))
