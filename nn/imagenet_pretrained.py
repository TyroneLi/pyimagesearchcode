from keras.applications import ResNet50, InceptionV3, Xception, VGG16, VGG19
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
ap.add_argument("-m", "--model", type=str, help="path to the pre-trained model")
args = vars(ap.parse_args())

models = {
	"vgg16":VGG16,
	"vgg19":VGG19,
	"inception":InceptionV3,
	"xception":Xception,
	"resnet":ResNet50
}

if args["model"] not in models.keys():
	raise AssertionError("the model you have not supplied yet.please check by command line.", 
		"be a key in the 'models' dict")
'''
VGG16, VGG19, and ResNet 接收 224 × 224 input images；Inception V3 and
Xception 接受 229× 229 inputs images
'''
inputShape = (224, 224)
preprocess = imagenet_utils.preprocess_input

if args["model"] in ("inception", "xception"):
	inputShape = (299, 299)
	preprocess = preprocess_input

print('[INFO] loading {}...'.format(args["model"]))
NetWork = models[args['model']]
model = NetWork(weights="imagenet")

print("[INFO] loading and pre-processing image...")
image = load_img(args['image'], target_size=inputShape)
image = img_to_array(image)

image = np.expand_dims(image, axis=0)
image = preprocess(image)

print("[INFO] classifying image with '{}'...".format(args["model"]))
preds = model.predict(image)
p = imagenet_utils.decode_predictions(preds)

for (i, (imagenetID, label, prob)) in enumerate(p[0]):
	print("{}. {}:{:.2f}%".format(i+1, label, prob*100))

original_image = cv2.imread(args["image"])
(iamgenetID, label, prob) = p[0][0]
cv2.putText(original_image, "Label:{}".format(label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#cv2.imshow("Classification", original_image)
#cv2.waitKey(0)
plt.imshow(original_image)
plt.grid(True)
plt.axis('off')
plt.show()