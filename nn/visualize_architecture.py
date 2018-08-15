from lenet import LeNet
from miniVGG import MiniVGGNet
from keras.utils import plot_model

model1 = LeNet.build(28, 28, 1, 10)
plot_model(model1, to_file="lenet-5-mnist.png", show_shapes=True)

model12 = LeNet.build(32, 32, 3, 10)
plot_model(model12, to_file="lenet-5-cifar10.png", show_shapes=True)

model13 = LeNet.build(32, 32, 3, 100)
plot_model(model13, to_file="lenet-5-cifar100.png", show_shapes=True)

model2 = MiniVGGNet.build(28, 28, 1, 10)
plot_model(model2, to_file="miniVGGNet-mnist.png", show_shapes=True)

model21 = MiniVGGNet.build(32, 32, 3, 10)
plot_model(model21, to_file="miniVGGNet-cifar10.png", show_shapes=True)

model22 = MiniVGGNet.build(32, 32, 3, 100)
plot_model(model22, to_file="miniVGGNet-cifar100.png", show_shapes=True)
