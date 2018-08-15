from lenet import LeNet
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np

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
opt = SGD(lr=0.005)
model = LeNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])
print(model.summary())
print("[INFO] training network Lenet-5")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=128, epochs=100, verbose=1)

model.save("./lenet-5-MNIST.hdf5")

print("[INFO] evaluating Lenet-5..")
preds = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1), 
	target_names=[str(x) for x in lb.classes_]))

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
plt.savefig("./lenet-5-loss_acc.png")	
