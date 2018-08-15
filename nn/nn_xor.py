from neuralnetwork import NeuralNetwork
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# 生成的数据
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or = np.array([[0], [1], [1], [1]])
y_and = np.array([[0], [0], [0], [1]])
y_xor = np.array([[0], [1], [1], [0]])
# 构造2-2-1结构的神经网络，2个节点输入层，2个节点的隐藏层，1个节点的输出层
nn = NeuralNetwork([2, 2, 1], alpha=0.5)
# 模型开始训练，更新得到最终不断迭代更新的weigh矩阵
losses = nn.fit(X, y_xor, epochs=2)
# 打印输出
for (x, target) in zip(X, y_xor):
	pred = nn.predict(x)[0][0]
	step = 1 if pred > 0.5 else 0
	print("[INFO] data-{}, ground_truth={}, pred={:.4f}, step={}"
		.format(x, target[0], pred, step))

print("X.shape\n", X.shape)
print("y_xor.shape\n", y_xor.shape)

# 可视化训练过程
plt.style.use("ggplot")
plt.figure()
plt.title("Data")
cm_dark = mpl.colors.ListedColormap(['g', 'b'])
plt.scatter(X[:, 0], X[:, 1], marker="o", c=y_xor.ravel(), cmap=cm_dark, s=80)
# print(testY)

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, len(losses)), losses)
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()

print("W\n", nn.W)