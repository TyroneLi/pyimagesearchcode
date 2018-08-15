import matplotlib
matplotlib.use("Agg")
from sklearn.model_selection import train_test_split
from sklearn import datasets
from keras import backend as K
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from miniVGG import MiniVGGNet
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from keras.datasets import cifar10
from keras.datasets import cifar100
import matplotlib.pyplot as plt
import numpy as np
import argparse

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

# 加载数据MNIST，然后归一化到【0,1】，同时使用75%做训练，25%做测试
print("[INFO] loading MNIST (full) dataset")
dataset = datasets.fetch_mldata("MNIST Original", data_home="/home/king/test/python/train/pyimagesearch/nn/data/")
data = dataset.data

if K.image_data_format() == "channels_first":
	data = data.reshape(data.shape[0], 1, 28, 28)
else:
	data = data.reshape(data.shape[0], 28, 28, 1)

(trainX, testX, trainY, testY) = train_test_split(data / 255.0, 
	dataset.target.astype("int"), test_size=0.25, random_state=42)
# 将label进行one-hot编码
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

print("[INFO] compiling model...")
callbacks = [LearningRateScheduler(step_decay)]
# opt = SGD(lr=0.01, decay=0.01 / 70, momentum=0.9, nesterov=True)
opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])
print(model.summary())
print("[INFO] training network miniVGG...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=70, 
	callbacks=callbacks, verbose=1)
# H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=70, verbose=1)

model.save(args["model"])

print("[INFO] evaluating miniVGG...")
preds = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1), 
	target_names=[str(x) for x in lb.classes_]))

# 保存可视化训练结果
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 70), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 70), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 70), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 70), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("# Epoch")
plt.ylabel("Loss/Accuracy without BatchNormalization")
plt.legend()
plt.savefig(args["output"])