import matplotlib
matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from miniGoogleNet import MiniGoogleNet
from trainingmonitor import TrainingMonitor
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.datasets import cifar10
import numpy as np
import argparse
import os

num_epochs = 70
init_lr = 2e-3

def poly_decay(epoch):
	maxEpochs = num_epochs
	baseLR = init_lr
	power = 1.0

	alpha = baseLR* (1 - (epoch / float(maxEpochs))) ** power
	return alpha


ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to output directory (logs, plots, etc.)")
ap.add_argument("-m", "--model", required=True, help="path to output model")
args = vars(ap.parse_args())

print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float")
testX = testX.astype("float")

mean = np.mean(trainX, axis=0)
trainX -= mean
testX -= mean

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

aug = ImageDataGenerator(width_shift_range=0.1, 
	height_shift_range=0.1, horizontal_flip=True, fill_mode="nearest")

figPath = os.path.sep.join([args["output"], "{}.png".format(os.getpid())])
jsonPath = os.path.sep.join([args["output"], "{}.json".format(os.getpid())])
callbacks = [TrainingMonitor(figPath, jsonPath=jsonPath), LearningRateScheduler(poly_decay)]

print("[INFO] compiling model...")
opt = SGD(lr=init_lr, decay=0.1 / 70, momentum=0.9)
# opt = SGD(lr=init_lr, momentum=0.9)
model = MiniGoogleNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] training network...")
model.fit_generator(aug.flow(trainX, trainY, batch_size=64), validation_data=(testX, testY), 
	steps_per_epoch=len(trainX) // 64, epochs=num_epochs, callbacks=callbacks)

print("[INFO] serializing network...")
model.save(args["model"])


