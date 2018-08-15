from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from simplepreprocessor import SimplePreprocessor
from simpledatasetloader import SimpleDatasetLoader
from imutils import paths
import argparse

if __name__ == '__main__':
	# 命令行参数设置
	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
	ap.add_argument("-k", "--neighbors", type=int, default=1, help="of nearest neighbors for classification")
	ap.add_argument("-j", "--jobs", type=int, help="of jobs for K-NN distance (-1 uses all variables cores)")
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

	print("[INFO] features matrix:{:.1f}MB".format(data.nbytes / (1024*1000.0)))
	# 对类别进行编码，比如dog-0,cat-1
	le = LabelEncoder()
	labels = le.fit_transform(labels)
	# print(labels)
	# 训练集70%用来训练，25%用来测试，其实也是相当于validation
	(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

	# print(trainX)
	# print(trainY)

	print("[INFO] evaluating K-NN classifier...")
	# 构建KNN分类器
	model = KNeighborsClassifier(n_neighbors=args["neighbors"], n_jobs=args["jobs"])
	# 模型训练
	model.fit(trainX, trainY)
	print(classification_report(testY, model.predict(testX), target_names=le.classes_))
