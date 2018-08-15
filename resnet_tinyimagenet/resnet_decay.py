import matplotlib
matplotlib.use("Agg")
from config import tiny_imagenet_config as config
from sklearn.preprocessing import LabelBinarizer
from imagetoarraypreprocessor import ImageToArrayPreprocessor
from simplepreprocessor import SimplePreprocessor
from meanpreprocessor import MeanPreprocessor
from epochcheckpoint import EpochCheckpoint
from trainingmonitor import TrainingMonitor
from hdf5datasetgenerator import HDF5DatasetGenerator
from resnet import ResNetPreActivation
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD
from keras.models import load_model
from keras.callbacks import LearningRateScheduler
import keras.backend as K
import argparse
import json
import os

# sys.setrecursionLimit(5000)

num_epochs = 100
init_lr = 1e-2

def poly_decay(epoch):
	maxEpochs = num_epochs
	baseLR = init_lr
	power = 2.0

	alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power

	return alpha

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to output model")
ap.add_argument("-o", "--output", required=True, help="path to output directory (logs plots etc)")
args = vars(ap.parse_args())

aug = ImageDataGenerator(rotation_range=18, zoom_range=0.15, width_shift_range=0.1, height_shift_range=0.1, 
	shear_range=0.15, horizontal_flip=True, fill_mode="nearest")

means = json.loads(open(config.DATASET_MEAN).read())

sp = SimplePreprocessor(64, 64)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, 64, 
	aug=aug, preprocessors=[sp, mp, iap], classes=config.NUM_CLASSES)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, 64, preprocessors=[sp, mp, iap], 
	classes=config.NUM_CLASSES)

figPath = os.path.sep.join([args["output"], "{}.png".format(os.getpid())])
jsonPath = os.path.sep.join([args["output"], "{}.json".format(os.getpid())])
callbacks = [TrainingMonitor(figPath, jsonPath=jsonPath), LearningRateScheduler(poly_decay)]

print("[INFO] compiling model...")
opt = SGD(lr=init_lr, momentum=0.9)
model = ResNetPreActivation.build(64, 64, 3, config.NUM_CLASSES, (3, 4, 6), (32, 64, 128, 256), reg=0.0005, dataset="tiny_imagenet")
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

model.fit_generator(trainGen.generator(), steps_per_epoch=trainGen.numImages // 64, 
	validation_data=valGen.generator(), validation_steps=valGen.numImages // 64, epochs=num_epochs, 
	max_queue_size=64 * 2, callbacks=callbacks, verbose=1)

print("[INFO] serializing network...")
model.save(args["model"])

trainGen.close()
valGen.close()