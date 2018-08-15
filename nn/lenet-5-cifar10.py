from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from shallownet import ShallowNet
from keras.optimizers import SGD
from lenet import LeNet
# from keras.datasets import cifar10
from keras.datasets import cifar100
import matplotlib.pyplot as plt
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to the output loss/accuracy plot")
ap.add_argument("-m", "--model", required=True, help="path to save train model")
args = vars(ap.parse_args())

print("[INFO] loading CIFAR-10 dataset")
((trainX, trainY), (testX, testY)) = cifar100.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)
# 标签0-9代表的类别string
labelNames = ['airplane', 'automobile', 'bird', 'cat', 
	'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

print("[INFO] compiling model...")
# opt = SGD(lr=0.007)
opt = SGD(lr=0.01, decay=0.01 / 100, momentum=0.9, nesterov=True)
model = LeNet.build(width=32, height=32, depth=3, classes=100)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])
print(model.summary())
print("[INFO] training network Lenet-5")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=128, epochs=100, verbose=1)

model.save(args["model"])

print("[INFO] evaluating Lenet-5..")
preds = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1), 
	target_names=labelNames))

# 保存可视化训练结果
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("# Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])