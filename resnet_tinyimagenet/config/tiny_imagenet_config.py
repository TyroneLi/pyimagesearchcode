
from os import path

TRAIN_IMAGES = "/home/king/test/python/train/pyimagesearch/deepergooglenet/datasets/tiny-imagenet-200/train"
VAL_IMAGES = "/home/king/test/python/train/pyimagesearch/deepergooglenet/datasets/tiny-imagenet-200/val/images"

VAL_MAPPINGS = "/home/king/test/python/train/pyimagesearch/deepergooglenet/datasets/tiny-imagenet-200/val/val_annotations.txt"

WORDNET_IDS = "/home/king/test/python/train/pyimagesearch/deepergooglenet/datasets/tiny-imagenet-200/wnids.txt"
WORD_LABELS = "/home/king/test/python/train/pyimagesearch/deepergooglenet/datasets/tiny-imagenet-200/words.txt"

NUM_CLASSES = 200
NUM_TEST_IMAGES = 50 * NUM_CLASSES

TRAIN_HDF5 = "/home/king/test/python/train/pyimagesearch/deepergooglenet/datasets/tiny-imagenet-200/train.hdf5"
VAL_HDF5 = "/home/king/test/python/train/pyimagesearch/deepergooglenet/datasets/tiny-imagenet-200/val.hdf5"
TEST_HDF5 = "/home/king/test/python/train/pyimagesearch/deepergooglenet/datasets/tiny-imagenet-200/test.hdf5"

DATASET_MEAN = "/home/king/test/python/train/pyimagesearch/deepergooglenet/tiny_imagenet_200_mean.json"

OUTPUT_PATH = "output"
MODEL_PATH = path.sep.join([OUTPUT_PATH, "checkpoints/resnet_tinyimagenet.hdf5"])
FIG_PATH = path.sep.join([OUTPUT_PATH, "resnet56_tinyimagenet.png"])
JSON_PATH = path.sep.join([OUTPUT_PATH, "resnet56_tinyimagenet.json"])
