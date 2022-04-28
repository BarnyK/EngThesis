from os import path

from data.file_handling import read_uint16png
from .utils import match_images_disparities
from torch.utils.data import random_split
from torch import Generator


def index_weather(root, split=0.2, validation_length=-1, **kwargs):
    if validation_length < 0 and split < 0 or split > 1:
        raise ValueError("split should be a float between 0 and 1")

    left_folder = path.join(root, "left-image-half-size")
    right_folder = path.join(root, "right-image-half-size")
    disp_folder = path.join(root, "disparity-map-half-size")

    data = match_images_disparities(left_folder, right_folder, disp_folder, "jpg")

    if validation_length > 0:
        train_length = len(data) - validation_length
    else:
        validation_length = int(len(data) * split)
        train_length = len(data) - validation_length

    trainset, testset = random_split(
        data, [train_length, validation_length], generator=Generator().manual_seed(1111)
    )
    trainset = sorted(trainset, key=lambda x: x[0])
    testset = sorted(testset, key=lambda x: x[0])

    return trainset, testset, read_uint16png


def combine_weathers(root, split, **kwargs):
    weathers = "sunny", "cloudy", "foggy", "rainy"
    paths = [path.join(root, w) for w in weathers]
    trainset = []
    testset = []
    for p in paths:
        train, test, _ = index_weather(p, split)
        trainset.extend(train)
        testset.extend(test)
    return trainset, testset, read_uint16png
