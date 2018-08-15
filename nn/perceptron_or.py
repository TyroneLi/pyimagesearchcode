from perceptron import Perceptron
import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or = np.array([[0], [1], [1], [1]])
y_and = np.array([[0], [0], [0], [1]])
y_xor = np.array([[1], [0], [0], [1]])

print("[INFO] training perceptron....")
p = Perceptron(X.shape[1], alpha=0.1)
p.fit(X, y_or, epochs=20)

print("[INFO] testing perceptron OR...")
for (x, target) in zip(X, y_or):
	pred = p.predict(x)
	print("[INFO] data={}, ground_truth={}, pred={}".format(x, target[0], pred))

print("[INFO] training perceptron AND....")
p = Perceptron(X.shape[1], alpha=0.1)
p.fit(X, y_and, epochs=20)

print("[INFO] testing perceptron AND...")
for (x, target) in zip(X, y_and):
	pred = p.predict(x)
	print("[INFO] data={}, ground_truth={}, pred={}".format(x, target[0], pred))

print("[INFO] training perceptron XOR....")
p = Perceptron(X.shape[1], alpha=0.1)
p.fit(X, y_xor, epochs=200)

print("[INFO] testing perceptron XOR...")
for (x, target) in zip(X, y_xor):
	pred = p.predict(x)
	print("[INFO] data={}, ground_truth={}, pred={}".format(x, target[0], pred))

print("X.shape\n", X.shape)
print("X.shape[0]\n", X.shape[0])
print("X.shape[1]\n", X.shape[1])
	