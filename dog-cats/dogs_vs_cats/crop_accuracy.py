from config import dogs_vs_cats_config as config
from imagetoarraypreprocessor import ImageToArrayPreprocessor
from simplepreprocessor import SimplePreprocessor
from meanpreprocessor import MeanPreprocessor
from croppreprocessor import CropPreprocessor
from hdf5datasetgenerator import HDF5DatasetGenerator
from ranked import rank5_accuracy
from keras.models import load_model
import numpy as np
import progressbar
import json

means = json.loads(open(config.DATASET_MEAN).read())

sp = SimplePreprocessor(227, 227)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
cp = CropPreprocessor(227, 227)
iap = ImageToArrayPreprocessor()

print("[INFO] loading model...")
model = load_model(config.MODEL_PATH)

print("[INFO] predicting on test data(no crop)...")
testGen = HDF5DatasetGenerator(config.TEST_HDF5, 64, preprocessors=[sp, mp, iap], classes=2)
predictions = model.predict_generator(testGen.generator(), steps=testGen.numImages // 64, max_queue_size=64 * 2)

(rank_1, _) = rank5_accuracy(predictions, testGen.db["labels"])
print("[INFO] rank-1:{:.2f}%".format(rank_1 * 100))
testGen.close()

testGen = HDF5DatasetGenerator(config.TEST_HDF5, 64, preprocessors=[mp], classes=2)
predictions = []

widgets = ["Evaluating:", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=testGen.numImages // 64, widgets=widgets).start()

for (i, (images, labels)) in enumerate(testGen.generator(passes=1)):
	for image in images:
		crops = cp.preprocess(image)
		crops = np.array([iap.preprocess(c) for c in crops], dtype="float32")

		preds = model.predict(crops)
		predictions.append(preds.mean(axis=0))
	pbar.update(i)

pbar.finish()
print("[INFO] predicting on testing data (with crops)...")
(rank1, _) = rank5_accuracy(predictions, testGen.db["labels"])
print("[INFO] rank-1:{:.2f}%".format(rank1 * 100))
testGen.close()

