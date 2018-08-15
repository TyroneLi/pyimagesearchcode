from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imagetoarraypreprocessor import ImageToArrayPreprocessor
from aspectawarepreprocessor import AspectAwarePreprocessor
from simpledatasetloader import SimpleDatasetLoader
from fcHeadNet import FCHeadNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.applications import VGG16
from keras.layers import Input
from keras.models import Model
from imutils import paths
import numpy as np
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--model", required=True, help="path to output model")
args = vars(ap.parse_args())

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, 
	shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]

aap = AspectAwarePreprocessor(224, 224)
iap = ImageToArrayPreprocessor()

sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# 开始进行fune tuning--network surgery 网络手术
baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

headModel = FCHeadNet.build(baseModel, len(classNames), 256)

model = Model(inputs=baseModel.input, output=headModel)

# 冻结除了最后一层新加进来的全连接层，其他所有的层均要被冻结
for layer in baseModel.layers:
	layer.trainable = False

# 开始定义网络，然后训练
print("[INFO] compiling model...")
opt = RMSprop(lr=0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# 开始训练新加入的全连接层， 训练一小epoch后，实现其真正学到一些特征而不再是纯碎的随机初始化值
# 在这一过程中其他的层都是被冻结的，也就是反向传播不存在
# 这一过程其实相当于初始化最后加进来的全连接层
print("[INFO] training head...")
model.fit_generator(aug.flow(trainX, trainY, batch_size=32), 
	validation_data=(testX, testY), epochs=25, steps_per_epoch=len(trainX) // 32, verbose=1)

print("[INFO] evaluating after initialization...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=classNames))

# 现在最后加进来的全连接层已经初始化完成，并且进行了一定程度上的
# 训练，现在开始解冻一些其他的层
for layer in baseModel.layers[15:]:
	layer.trainable = True

# 重新训练
print("[INFO] re-compiling model...")
opt = SGD(lr=0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] fine-tuning training model...")
model.fit_generator(aug.flow(trainX, trainY, batch_size=32), 
	validation_data=(testX, testY), epochs=100, steps_per_epoch=len(trainX) // 32, verbose=1)

print("[INFO] evaluating after fine-tuning...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=classNames))

print("[INFO] serializing model...")
model.save(args["model"])

