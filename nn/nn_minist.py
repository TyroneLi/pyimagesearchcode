from neuralnetwork import NeuralNetwork
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
# 从sklearn包中导入数据MNIST，其实是MNIST数据集的缩小版，仅包含1797张images
print("[INFO] loading mnist dataset...")
digits = datasets.load_digits()
data = digits.data.astype("float")
# print(data)
# 归一化到（0， 1）
data = (data - data.min()) / (data.max() - data.min())
print("[INFO] samples:{}, dim:{}".format(data.shape[0], data.shape[1]))
# print(data)
# 75%做训练数据集，25%做测试数据集
(trainX, testX, trainY, testY) = train_test_split(data, digits.target, test_size=0.25)

print("trainY:\n", trainY)
print("testY:\n", testY)
# 将标签值向量化，即是one-hot编码，如0--[1,0,0,0,0,0,0,0,0,0],1--[0,1,0,0,0,0,0,0,0,0],9--[0,0,0,0,0,0,0,0,0,1]
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

print("Vectorize trainY:\n", trainY)
print("trainY[0]\n", testY[0])
print("Vectroize testY:\n", testY)
# 定义网络结构64-32-32-16-10,64表示输入层有64个nodes(因为8x8=64)，输出层有10个nodes(10个数值0-9输出)
print("[INFO] training network...")
nn = NeuralNetwork([trainX.shape[1], 32, 32, 16, 10])
print("[INFO] {}".format(nn))
# print("trainX.shape[0]:\n", trainX.shape[0])
# print("trainX.shape:\n", trainX.shape)

print("trainX.shape\n", trainX.shape)
print("testY.shape\n", testY.shape)
# 训练模型
losses = nn.fit(trainX, trainY, epochs=5000)
# 预测，并生成报告
print("[INFO] evaluating network...")
predictions = nn.predict(testX)
predictions = predictions.argmax(axis=1)
print(classification_report(testY.argmax(axis=1), predictions))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, len(losses)), losses)
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()

print("W\n", nn.W)
