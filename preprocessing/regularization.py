from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from simplepreprocessor import SimplePreprocessor
from simpledatasetloader import SimpleDatasetLoader
from imutils import paths
import argparse

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
	args = vars(ap.parse_args())

	print("[INFO] loading images...")
	# 加载数据集的文件路径
	imagePaths = list(paths.list_images(args["dataset"]))
	# 对数据集文件夹下的图片进行预处理，统一到32x32的尺寸
	sp = SimplePreprocessor(32, 32)
	sdl = SimpleDatasetLoader(preprocessors=[sp])
	# 从RGB三颜色通道flat到1维矩阵
	(data, labels) = sdl.load(imagePaths, verbose=500)
	data = data.reshape((data.shape[0], 3072))

	le = LabelEncoder()
	labels = le.fit_transform(labels)

	(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=5)

	for r in(None, "l1", "l2"):
		print("[INFO] training model with '{}' penalty".format(r))
		model = SGDClassifier(loss="log", penalty=r, max_iter=50, learning_rate="constant", eta0=0.001, random_state=42)
		model.fit(trainX, trainY)

		acc = model.score(testX, testY)
		print("[INFO] '{}' penalty accuracy:{:.3f}%".format(r, acc * 100))

if __name__ == '__main__':
	main()