import matplotlib
matplotlib.use("Agg")
from sklearn.preprocessing import LabelBinarizer
from resnet import ResNetPreActivation
from epochcheckpoint import EpochCheckpoint
from trainingmonitor import TrainingMonitor
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.models import load_model
from keras.callbacks import LearningRateScheduler
import keras.backend as K
import numpy as np
import argparse
import sys

# sys.setrecursionLimit(5000)

num_epochs = 100
init_lr = 1e-1

def poly_decay(epoch):
	maxEpochs = num_epochs
	baseLR = init_lr
	power = 1.0

	alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power

	return alpha

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True, help="path to output checkpoints dorectory")
# ap.add_argument("-sm", "--save-model", required=True, help="path to save model")
ap.add_argument("-m", "--model", type=str, help="path to output model checkpoints to load")
ap.add_argument("-s", "--start_epoch", type=int, default=0, help="epoch to restart training at #")
args = vars(ap.parse_args())

print("[INFO] loading CIFAR-10 Dataset...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float")
testX = testX.astype("float")

mean = np.mean(trainX, axis=0)
trainX -= mean
# mean = np.mean(testX, axis=0)
testX -= mean

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

aug = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, fill_mode="nearest")

if args["model"] is None:
	print("[INFO] compiling model...")
	opt = SGD(lr=1e-1)
	model = ResNetPreActivation.build(32, 32, 3, 10, (9, 9, 9), (64, 64, 128, 256), reg=0.0005)
	model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
else:
	print("[INFO] loading {}...".format(args["model"]))
	model = load_model(args["model"])

	print("[INFO] old learning rate:{}".format(K.get_value(model.optimizer.lr)))
	K.set_value(model.optimizer.lr, 1e-3)
	print("[INFO] new learning rate:{}".format(K.get_value(model.optimizer.lr)))

callbacks = [
	EpochCheckpoint(args["checkpoints"], every=5, startAt=args["start_epoch"]),
	TrainingMonitor("output/resnet56_cifar10.png", jsonPath="output/resnet56_cifar10.json", startAt=args["start_epoch"])
]

print("[INFO] training network...")
print("trainX.shape:\n", trainX.shape)
print("testX.shape:\n", testX.shape)
print("trainY.shape:\n", trainY.shape)
print("testY.shape:\n", testY.shape)
model.fit_generator(aug.flow(trainX, trainY, batch_size=128), validation_data=(testX, testY), 
	steps_per_epoch=len(trainX) // 128, epochs=10, callbacks=callbacks, verbose=1)

# print("[INFO] serializing network...")
# model.save(args["save-model"])