from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2
# 自定义卷积操作
def convolve(image, K):
	(iH, iW) = image.shape[:2]
	(kH, kW) = K.shape[:2]
	# 边界扩充
	pad = (kW - 1) // 2
	image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
	output = np.zeros((iH, iW), dtype="float")
	# 卷积操作
	for y in np.arange(pad, iH+pad):
		for x in np.arange(pad, iW+pad):
			roi = image[y-pad:y+pad+1, x-pad:x+pad+1]

			k = (roi * K).sum()

			output[y-pad, x-pad] = k
	# 将卷积运算后的像素重新映射到[0, 255]正常范围内
	output = rescale_intensity(output, in_range=(0, 255))
	output = (output * 255).astype('uint8')

	return output
# 命令行参数运行
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to the input image')
args = vars(ap.parse_args())
# 卷积核
smallBlur = np.ones((7, 7), dtype="float")*(1.0 / (7 * 7))
largeBlur = np.ones((21, 21), dtype="float")*(1.0 / (21 * 21))
# 锐化
sharpen = np.array((
	[0, -1, 0],
	[-1, 5, -1],
	[0, -1, 0]
	), dtype="int")
# 拉普拉斯算子
laplacian = np.array((
	[0, 1, 0],
	[1, -4, 1],
	[0, 1, 0]
	), dtype="int")
# sobel算子
sobelX = np.array((
	[-1, 0, 1],
	[-2, 0, 2],
	[-1, 0, 1]
	), dtype="int")

sobelY = np.array((
	[-1, -2, -1],
	[0, 0, 0],
	[1, 2, 1]
	), dtype="int")

emboss = np.array((
	[-2, -1, 0],
	[-1, 1, 1],
	[0, 1, 2]
	), dtype="int")
# 聚集成tuple
kernelBank = (
	("small_Blur", smallBlur),
	("large_Blur", largeBlur),
	("sharpen", sharpen),
	("laplacian", laplacian),
	("sobel_x", sobelX),
	("sobel_y", sobelY),
	("emboss", emboss))
# 加载图像，并且转换成灰度图
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 进行卷积操作，并且和opencv的filter2D卷积函数效果对比
for (kernelName, K) in kernelBank:
	print("[INFO] applying {} kernel".format(kernelName))
	convolveOutput = convolve(gray, K)
	opencvOutput = cv2.filter2D(gray, -1, K)
	# 可视化
	cv2.imshow("Original", gray)
	cv2.imshow("{} - convole".format(kernelName), convolveOutput)
	cv2.imshow("{} - opencv".format(kernelName), opencvOutput)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
