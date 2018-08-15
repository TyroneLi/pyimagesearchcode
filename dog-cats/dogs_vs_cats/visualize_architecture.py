from alexnet import Alexnet
from keras.utils import plot_model
from keras.applications import ResNet50
from keras.applications import VGG16

alexnet_model = Alexnet.build(227, 227, 3, 2)
plot_model(alexnet_model, to_file="alexnet_model-structure.png", show_shapes=True)

resnet_model = ResNet50(weights="imagenet", include_top=False)
plot_model(resnet_model, to_file="resnet50_model-structure.png", show_shapes=True)

vgg_model = VGG16(weights="imagenet", include_top=False)
plot_model(vgg_model, to_file="vgg16_model-structure.png", show_shapes=True)
