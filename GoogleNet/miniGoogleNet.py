from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, AveragePooling2D, MaxPooling2D
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, Input,concatenate
from keras.models import Model
from keras import backend as K

class MiniGoogleNet:
	@staticmethod
	def conv_module(x, K, kX, kY, stride, chanDim, padding="same"):
		x = Conv2D(K, (kX, kY), strides=stride, padding=padding)(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = Activation("relu")(x)

		return x
	def inception_module(x, numK1x1, numK3x3, chanDim):
		conv_1x1 = MiniGoogleNet.conv_module(x, numK1x1, 1, 1, (1, 1), chanDim)
		conv_3x3 = MiniGoogleNet.conv_module(x, numK3x3, 3, 3, (1, 1), chanDim)
		x = concatenate([conv_1x1, conv_3x3], axis=chanDim)

		return  x

	@staticmethod
	def downsample_module(x, K, chanDim):
		conv_3x3 = MiniGoogleNet.conv_module(x, K, 3, 3, (2, 2), chanDim, padding="valid")
		pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
		x = concatenate([conv_3x3, pool], axis=chanDim)

		return x
	@staticmethod
	def build(width, height, depth, classes):
		inputShape = (height, width, depth)
		chanDim = -1

		if K.image_data_format() == "channel_first":
			inputShape = (depth, height, width)
			chanDim = 1

		inputs = Input(shape=inputShape)
		x = MiniGoogleNet.conv_module(inputs, 96, 3, 3, (1, 1), chanDim)

		x = MiniGoogleNet.inception_module(x, 32, 32, chanDim)
		x = MiniGoogleNet.inception_module(x, 32, 48, chanDim)
		x = MiniGoogleNet.downsample_module(x, 80, chanDim)

		x = MiniGoogleNet.inception_module(x, 112, 48, chanDim)
		x = MiniGoogleNet.inception_module(x, 96, 64, chanDim)
		x = MiniGoogleNet.inception_module(x, 80, 80, chanDim)
		x = MiniGoogleNet.inception_module(x, 48, 96, chanDim)
		x = MiniGoogleNet.downsample_module(x, 96, chanDim)

		x = MiniGoogleNet.inception_module(x, 176, 160, chanDim)
		x = MiniGoogleNet.inception_module(x, 176, 160, chanDim)
		x = AveragePooling2D((7, 7))(x)
		x = Dropout(0.5)(x)

		x = Flatten()(x)
		x = Dense(classes)(x)
		x = Activation("softmax")(x)

		model = Model(inputs, x, name="GooleNet")

		return model
