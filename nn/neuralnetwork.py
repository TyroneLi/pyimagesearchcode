import numpy as np
# 将完整的神经网络结构定义成类
class NeuralNetwork:
	# 初始化，构造函数
	def __init__(self, layers, alpha=0.1):
		self.W = []
		self.layers = layers
		self.alpha = alpha
		
		# 除了最后两层网络外，其他的都初始化Weight
		for i in np.arange(0, len(layers) - 2):
			# 先初始化常规的weights矩阵
			w = np.random.randn(layers[i] + 1, layers[i+1] + 1)
			# 归一化
			self.W.append(w / np.sqrt(layers[i]))
			# print("W without bias trick:\n", self.W)
			# 使用bias trick也就是在W矩阵最后一列加入新的一列作为bias然后weight和bias合并为一个大W矩阵
			# biases可以作为学习参数进行学习
		w= np.random.randn(layers[-2] + 1, layers[-1])
		# 归一化
		self.W.append(w / np.sqrt(layers[-2]))
		# print("W with bias trick:\n", self.W)
		
	# 重载python的magic函数
	def __repr__(self):
		return "NeuralNetwork:{}".format("-".join(str(l) for l in self.layers))

	def sigmoid(self, x):
		return 1.0 / (1 + np.exp(-x))
	# 对sigmoid函数求导
	def sigmoid_deriv(self, x):
		'''
		y = 1.0 / (1 + np.exp(-x))
		return y * (1 - y)
		'''
		return x * (1 - x)

	def fit(self, X, y, epochs=1000, displayUpdate=100):
		X = np.c_[X, np.ones((X.shape[0]))]
		losses = []
		# 根据每一层网络进行反向传播，然后更新W
		for epoch in np.arange(0, epochs):
			for (x, target) in zip(X, y):
				self.fit_partial(x, target)
			# 控制显示，并且加入loss
			if epoch == 0 or (epoch + 1) % displayUpdate == 0:
				loss = self.calculate_loss(X, y)
				losses.append(loss)
				print("[INFO] epoch={}, loss={:.7f}".format(epoch + 1, loss))
		return losses
	# 链式求导
	def fit_partial(self, x, y):
		A = [np.atleast_2d(x)]

		for layer in np.arange(0, len(self.W)):
			# print("A[layer].shape\n", A[layer].shape)
			# print("self.W[layer].shape\n", self.W[layer].shape)
			net = A[layer].dot(self.W[layer])
			# net = np.dot(A[layer], self.W[layer])
			# print("net.shape\n", net.shape)
			out = self.sigmoid(net)

			A.append(out)

		# backprogation algorithm
		error = A[-1] - y

		D = [error * self.sigmoid_deriv(A[-1])]

		for layer in np.arange(len(A) - 2, 0, -1):
			delta = D[-1].dot(self.W[layer].T)
			delta = delta * self.sigmoid_deriv(A[layer])
			D.append(delta)

		D = D[::-1]
		# 更新权值W
		for layer in np.arange(0, len(self.W)):
			self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])
	# 预测
	def predict(self, X, addBias=True):
		p = np.atleast_2d(X)
		# 是否加入偏置
		if addBias:
			p = np.c_[p, np.ones((p.shape[0]))]
		# 正常的前向传播得到预测的输出值
		for layer in np.arange(0, len(self.W)):
			p = self.sigmoid(np.dot(p, self.W[layer]))

		return p
	# 计算loss，就是计算MSE
	def calculate_loss(self, X, targets):
		targets = np.atleast_2d(targets)
		predictions = self.predict(X, addBias=False)
		loss = 0.5 * np.sum((predictions - targets) ** 2)

		return loss


if __name__ == '__main__':
	nn = NeuralNetwork([2, 2, 1])
	print(nn)
