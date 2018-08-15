import matplotlib
matplotlib.use("Agg")

from config import dogs_vs_cats_config as config
from imagetoarraypreprocessor import ImageToArrayPreprocessor
from simplepreprocessor import SimplePreprocessor
from patchpreprocessor import PatchPreprocessor
from meanpreprocessor import MeanPreprocessor
from trainingmonitor import TrainingMonitor
from hdf5datasetgenerator import HDF5DatasetGenerator
from alexnet import Alexnet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import json
import os

aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, 
	height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, fill_mode="nearest")

means = json.loads(open(config.DATASET_MEAN).read())
sp = SimplePreprocessor(227, 227)
pp = PatchPreprocessor(227, 227)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()

trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, 128, aug=aug, preprocessors=[pp, mp, iap], classes=2)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, 128, aug=aug, preprocessors=[sp, mp, iap], classes=2)

print("[INFO] compiling model...")
opt = Adam(lr=1e-3)
model = Alexnet.build(width=227, height=227, depth=3, classes=2, reg=0.0002)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

path = os.path.sep.join([config.OUTPUT_PATH, "{}.png".format(os.getpid())])
callbacks = [TrainingMonitor(path)]

model.fit_generator(trainGen.generator(), steps_per_epoch=trainGen.numImages // 128, 
	validation_data=valGen.generator(), validation_steps=valGen.numImages // 128, epochs=75, 
	max_queue_size=128*2, callbacks=callbacks, verbose=1)

print("[INFO] serializing model...")
model.save(config.MODEL_PATH, overwrite=True)

trainGen.close()
valGen.close()

