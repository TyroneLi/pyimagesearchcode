from config import tiny_imagenet_config as config
from imagetoarraypreprocessor import ImageToArrayPreprocessor
from simplepreprocessor import SimplePreprocessor
from meanpreprocessor import MeanPreprocessor
from ranked import rank5_accuracy
from hdf5datasetgenerator import HDF5DatasetGenerator
from keras.models import load_model
import json

means = json.loads(open(config.DATASET_MEAN).read())

sp = SimplePreprocessor(64, 64)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()

testGen = HDF5DatasetGenerator(config.TRAIN_HDF5, 64, preprocessors=[sp, mp, iap], classes=config.NUM_CLASSES)

print("[INFO] loading model...")
model = load_model(config.MODEL_PATH)

print("[INFO] predictiong on test data...")
preds = model.predict_generator(testGen.generator(), steps=testGen.numImages//64, max_queue_size=64*2)

(rank_1, rank_5) = rank5_accuracy(preds, testGen.db["labels"])
print("[INFO] rank-1:{:.2f}".format(rank_1 * 100))
print("[INFO] rank-5:{:.2f}".format(rank_5 * 100))

testGen.close()