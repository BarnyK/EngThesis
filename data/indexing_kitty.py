from os import path
from .utils import match_images_disparities
from torch.utils.data import random_split
from torch import Generator


def index_kitti2012(
    root, occlussion=True, split=0.2, colored=True, validation_length=-1, **kwargs
):
    disp_folder = "disp_occ"
    if not occlussion:
        disp_folder = "disp_noc"
    if colored:
        return __index_kitti(
            root, "colored_0", "colored_1", disp_folder, "png", split, validation_length
        )
    return __index_kitti(
        root, "image_0", "image_1", disp_folder, "png", split, validation_length
    )


def index_kitti2015(root, occlussion=True, split=0.2, validation_length=-1, **kwargs):
    disp_folder = "disp_occ_0"
    if not occlussion:
        disp_folder = "disp_noc_0"
    train, test =  __index_kitti(
        root, "image_2", "image_3", disp_folder, "png", split, validation_length
    )
    xd = sorted(train+test)
    train  = xd

    test = [train[i] for i in range(len(train)) if i in [1,3,6,20,26,35,38,41,43,44,49,60,67,70,81,84,89,97,109,119,122,123,129,130,132,134,141,144,152,158,159,165,171,174,179,182,186,187,196]]
    train = [train[i] for i in range(len(train)) if i not in [1,3,6,20,26,35,38,41,43,44,49,60,67,70,81,84,89,97,109,119,122,123,129,130,132,134,141,144,152,158,159,165,171,174,179,182,186,187,196]]
    return train, test

def __index_kitti(
    root,
    left_folder,
    right_folder,
    disp_folder,
    input_extension="png",
    split=0.2,
    validation_length=-1,
):
    if validation_length < 0 and split < 0 or split > 1:
        raise ValueError("split should be a float between 0 and 1")

    left = path.join(root, "training", left_folder)
    right = path.join(root, "training", right_folder)
    disparity = path.join(root, "training", disp_folder)
    data = match_images_disparities(left, right, disparity, input_extension)

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

    return trainset, testset


def combine_kitti(root, occlussion=True, **kwargs):
    kitti2012_folder = path.join(root, "data_stereo_flow")
    kitti2015_folder = path.join(root, "data_scene_flow")
    kitti2012, kitti2012_test = index_kitti2012(
        kitti2012_folder, occlussion, validation_length=14
    )
    kitti2015, kitti2015_test = index_kitti2015(
        kitti2015_folder, occlussion, validation_length=20
    )
    trainset = kitti2012 + kitti2015
    testset = kitti2012_test + kitti2015_test
    return trainset, testset
