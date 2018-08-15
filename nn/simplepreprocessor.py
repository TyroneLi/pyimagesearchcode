import cv2

class SimplePreprocessor:
	def __init__(self, width, height, inter=cv2.INTER_AREA):
		self.width = width
		self.height = height
		self.inter = inter
	# 设置输入进来的图片统一到一个尺寸
	def preprocess(self, image):
		return cv2.resize(image, (self.width, self.height), interpolation=self.inter)


if __name__ == '__main__':

	s = SimplePreprocessor(300, 400)
	img = cv2.imread('beauty.jpg')
	# print(img)
	cv2.imshow('src', img)
	cv2.imshow("resize", s.preprocess(img))

	cv2.waitKey(0)
	# cv2.destroyallWindows()