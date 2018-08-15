import matplotlib
matplotlib.use("Agg")

from config import tiny_imagenet_config as config
from imagetoarraypreprocessor import ImageToArrayPreprocessor
from simplepreprocessor import SimplePreprocessor
from meanpreprocessor import MeanPreprocessor
from epochcheckpoint import EpochCheckpoint
from trainingmonitor import TrainingMonitor
from hdf5datasetgenerator import HDF5DatasetGenerator
from deepergooglenet import DeeperGoogleNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD
from keras.models import load_model
import keras.backend as K
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True, help="path to output checkpoint directory")
ap.add_argument("-m", "--models", type=str, help="path to specific model checkpoint to load")
ap.add_argument("-s", "--start_epoch", type=int, default=0, help="epoch to restart training at")
args = vars(ap.parse_args())

aug = ImageDataGenerator(rotation_range=18, zoom_range=0.15, width_shift_range=0.2, 
	height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, fill_mode="nearest")

means = json.loads(open(config.DATASET_MEAN).read())

sp = SimplePreprocessor(64, 64)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()

trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, 64, 
	aug=aug, preprocessors=[sp, mp, iap], classes=config.NUM_CLASSES)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, 64, preprocessors=[sp, mp, iap], 
	classes=config.NUM_CLASSES)

if args["models"] is None:
	print("[INFO] compiling model...")
	model = DeeperGoogleNet.build(width=64, height=64, depth=3, classes=config.NUM_CLASSES, reg=0.0002)
	# opt = Adam(1e-2)
	opt = SGD(1e-2)
	model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
else:
	print("[INFO] loading {}...".format(args["models"]))
	model = load_model(args["models"])

	print("[INFO] old learning rate:{}".format(K.get_value(model.optimizer.lr)))
	K.set_value(model.optimizer.lr, 1e-4)
	print("[INFO] new learning rate:{}".format(K.get_value(model.optimizer.lr)))

callbacks = [
	EpochCheckpoint(args["checkpoints"], every=5, startAt=args["start_epoch"]),
	TrainingMonitor(config.FIG_PATH, jsonPath=config.JSON_PATH, startAt=args["start_epoch"])
]

model.fit_generator(trainGen.generator(), steps_per_epoch=trainGen.numImages // 64, 
	validation_data=valGen.generator(), validation_steps=valGen.numImages // 64, epochs=30, 
	max_queue_size=64 * 2, callbacks=callbacks, verbose=1)

trainGen.close()
valGen.close()
