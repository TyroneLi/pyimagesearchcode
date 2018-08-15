import numpy as np
import cv2
import os

from ..preprocessing.simplepreprocessor import SimplePreprocessor

class SimpleDatasetLoader:
	def __init__(self, preprocessors=None):
		self.preprocessors = preprocessors

		if self.preprocessors is None:
			self.preprocessors = []

	def load(self, imagePaths, verbose=-1):
		data_cat = []
		data_dog = []
		labels_dog = []
		labels_cat = []

		for file_name in os.listdir(imagePaths):
			file_path = os.path.join(imagePaths, file_name)
			image = cv2.imread(file_path)
			if self.preprocessors is not None:
				s = simplepreprocessor.SimplePreprocessor()
				image = s.preprocess(image)
			# label = imagePath.split(os.path.sep)[-3]
			label = file_path.split('.')[-3].split('/')[-1]
			# print(label)
			if label == 'dog':
				data_dog.append(image)
				labels_dog.append(label)
			else:
				data_cat.append(image)
				labels_cat.append(label)
			print(data_cat)
			print(data_dog)
			print(labels_cat)
			print(labels_dog)


if __name__ == '__main__':
	s = SimpleDatasetLoader()
	imagePaths = '/home/king/test/python/train/data/all/train'
	s.load(imagePaths)