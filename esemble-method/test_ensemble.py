from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.models import load_model
from keras.datasets import cifar10
import numpy as np
import argparse
import glob
import os

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to output models directory")
args = vars(ap.parse_args())

(testX, testY) = cifar10.load_data()[1]
testX = testX.astype("float") / 255.0

# 标签0-9代表的类别string
labelNames = ['airplane', 'automobile', 'bird', 'cat', 
	'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
'''
labelNames = [
	'mammals beaver', 'dolphin', 'otter', 'seal', 'whale', 'aquarium fish', 'flatfish', 'ray', 'shark', 'trout',
	'orchids', 'poppies', 'roses', 'sunflowers', 'tulips', 'containers bottles', 'bowls', 'cans', 'cups', 'plates',
	'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers', 'clock', 'computer keyboard', 'lamp', 'telephone', 'television',
	'furniture bed', 'chair', 'couch', 'table', 'wardrobe', 'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach', 
	'bear', 'leopard', 'lion', 'tiger', 'wolf', 'bridge', 'castle', 'house', 'road', 'skyscraper',
	'cloud', 'forest', 'mountain', 'plain', 'sea', 'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
	'fox', 'porcupine', 'possum', 'raccoon', 'skunk', 'crab', 'lobster', 'snail', 'spider', 'worm',
	'baby', 'boy', 'girl', 'man', 'woman', 'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
	'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel', 'maple', 'oak', 'palm', 'pine', 'willow',
	'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train', 'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor'
]
'''

lb = LabelBinarizer()
testY = lb.fit_transform(testY)

modelPaths = os.path.sep.join([args["model"], "*.model"])
print(modelPaths)
modelPaths = list(glob.glob(modelPaths))
print(modelPaths)
models = []

for (i, modelPath) in enumerate(modelPaths):
	print("[INFO] loading model {}/{}".format(i+1, len(modelPaths)))
	print(modelPath)
	models.append(load_model(modelPath))

models.append(load_model("./models/model_2.model"))

print("[INFO] evaluating ensembles...")
predictions = []

for model in models:
	predictions.append(model.predict(testX, batch_size=64))

predictions = np.average(predictions, axis=0)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))

