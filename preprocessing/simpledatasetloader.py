import numpy as np
import cv2
import os
'''
class SimplePreprocessor:
	def __init__(self, width, height, inter=cv2.INTER_AREA):
		self.width = width
		self.height = height
		self.inter = inter
	@staticmethod
	def preprocess(self, image):
		return cv2.resize(image, (self.width, self.height), interpolation=self.inter)
'''

class SimpleDatasetLoader:
	def __init__(self, preprocessors=None):
		self.preprocessors = preprocessors

		if self.preprocessors is None:
			self.preprocessors = []

	def load(self, imagePaths, verbose=-1):
		'''
		data_cat = []
		data_dog = []
		labels_dog = []
		labels_cat = []
		'''
		# 接收图像数据(像素值)和相对应的label
		data = []
		labels = []

		for (i, imagePath) in enumerate(imagePaths):
			image = cv2.imread(imagePath)
			# 根据文件夹的分类进行获取label
			# 文件夹应该对应
			# dog_...  cat_...
			label = imagePath.split(os.path.sep)[-2]
			if self.preprocessors is not None:
				for p in self.preprocessors:
					image = p.preprocess(image)
			data.append(image)
			labels.append(label)
			# 每处理500张就输出信息
			if verbose > 0 and i > 0 and (i + 1)%verbose == 0:
				print('[INFO] processed {}/{}'.format(i+1, len(imagePaths)))

		# print(data_cat)
		# print(data_dog)
		# print(labels_cat)
		# print(labels_dog)

		return (np.array(data), np.array(labels))


if __name__ == '__main__':
	s = SimpleDatasetLoader()
	imagePaths = '/home/king/test/python/train/data/all/train'
	s.load(imagePaths)