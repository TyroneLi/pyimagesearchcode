from lenet import LeNet
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from keras import backend as K
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to load train model")
args = vars(ap.parse_args())

# 加载数据MNIST，然后归一化到【0,1】，同时使用75%做训练，25%做测试
print("[INFO] loading MNIST (full) dataset")
dataset = datasets.fetch_mldata("MNIST Original", data_home="/home/king/test/python/train/pyimagesearch/nn/data/")
data = dataset.data

if K.image_data_format() == "channels_first":
	data = data.reshape(data.shape[0], 1, 28, 28)
else:
	data = data.reshape(data.shape[0], 28, 28, 1)

(trainX, testX, trainY, testY) = train_test_split(data / 255.0, 
	dataset.target.astype("int"), test_size=0.005, random_state=42)
# 将label进行one-hot编码
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

print("[INFO] loading pre-trained network...")
model = load_model(args["model"])

print("[INFO] evaluating Lenet-5..")
preds = model.predict(testX, batch_size=128).argmax(axis=1)
print("predictions:\n", preds)
trueLabel = []
for i in range(len(testY)):
	for j in range(len(testY[i])):
		if testY[i][j] != 0:
			trueLabel.append(j)

print("ground truth:\n", trueLabel)

print("find wrong predictions:\n")
for i in range(len(trueLabel)):
	if trueLabel[i] != preds[i]:
		print("trueLabel:{}".format(trueLabel[i]))
		print("preds:{}".format(preds[i]))

