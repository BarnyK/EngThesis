from os import listdir, path

from torch import Generator
from torch.utils.data import random_split

from data.file_handling import read_pfm

from .utils import check_paths_exist, match_images_disparities


def index_driving(
    root_images,
    root_disparity,
    webp=True,
    disparity_side="left",
    split=0.2,
    validation_length=-1,
    **kwargs,
):
    if disparity_side not in ("left", "right"):
        raise ValueError("disparity_side should be either 'left' or 'right'")
    if validation_length < 0 and (split < 0 or split > 1):
        raise ValueError("split should be a float between 0 and 1")
    if not root_images:
        raise ValueError("specified image root path is None or doesn't exist")
    if not root_images:
        raise ValueError("specified disparity root path is None or doesn't exist")
    maindir = "frames_cleanpass_webp"
    extension = "webp"
    if not webp:
        maindir = "frames_cleanpass"
        extension = "png"

    internal_paths = (
        path.join("15mm_focallength", "scene_backwards", "fast"),
        path.join("15mm_focallength", "scene_backwards", "slow"),
        path.join("15mm_focallength", "scene_forwards", "fast"),
        path.join("15mm_focallength", "scene_forwards", "slow"),
        path.join("35mm_focallength", "scene_backwards", "fast"),
        path.join("35mm_focallength", "scene_backwards", "slow"),
        path.join("35mm_focallength", "scene_forwards", "fast"),
        path.join("35mm_focallength", "scene_forwards", "slow"),
    )

    data = []
    for p in internal_paths:
        left = path.join(root_images, maindir, p, "left")
        right = path.join(root_images, maindir, p, "right")
        disparity = path.join(root_disparity, "disparity", p, disparity_side)

        triplets = match_images_disparities(left, right, disparity, extension)
        data.extend(triplets)

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

    return trainset, testset, read_pfm


def index_flyingthings(
    root_images, root_disparity, webp=True, disparity_side="left", **kwargs
):
    if disparity_side not in ("left", "right"):
        raise ValueError("disparity_side should be either 'left' or 'right'")
    if not root_images:
        raise ValueError("specified image root path is None or doesn't exist")
    if not root_images:
        raise ValueError("specified disparity root path is None or doesn't exist")

    maindir = "frames_cleanpass_webp"
    extension = "webp"
    if not webp:
        maindir = "frames_cleanpass"
        extension = "png"

    def __index_flyingthings(subfolder1):
        data = []
        for subfolder2 in ["A", "B", "C"]:
            folder = path.join(root_disparity, "disparity", subfolder1, subfolder2)
            subfolders3 = [path.join(folder, sf) for sf in listdir(folder)]
            subfolders3 = [f for f in subfolders3 if path.isdir(f)]
            for sf in listdir(folder):
                img_path = path.join(root_images, maindir, subfolder1, subfolder2, sf)
                left = path.join(img_path, "left")
                right = path.join(img_path, "right")
                disparity = path.join(folder, sf, disparity_side)
                triplets = match_images_disparities(left, right, disparity, extension)
                data.extend(triplets)
        return data

    train_data = __index_flyingthings("TRAIN")
    test_data = __index_flyingthings("TEST")

    return train_data, test_data, read_pfm


def index_monkaa(
    root_images,
    root_disparity,
    webp=True,
    disparity_side="left",
    split=0.8,
    validation_length=-1,
    **kwargs,
):
    if disparity_side not in ("left", "right"):
        raise ValueError("disparity_side should be either 'left' or 'right'")
    if validation_length < 0 and (split < 0 or split > 1):
        raise ValueError("split should be a float between 0 and 1")
    if not root_images:
        raise ValueError("specified image root path is None or doesn't exist")
    if not root_images:
        raise ValueError("specified disparity root path is None or doesn't exist")

    maindir = "frames_cleanpass_webp"
    extension = "webp"
    if not webp:
        maindir = "frames_cleanpass"
        extension = "png"

    data = []
    for subfolder in listdir(path.join(root_images, maindir)):
        left = path.join(root_images, maindir, subfolder, "left")
        right = path.join(root_images, maindir, subfolder, "right")
        disparity = path.join(root_disparity, "disparity", subfolder, disparity_side)
        data.extend(match_images_disparities(left, right, disparity, extension))

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

    return trainset, testset, read_pfm


def combine_sceneflow(root, webp=True, disparity_side="left", **kwargs):
    if root is None:
        raise ValueError("root folder is not set")
    if not path.exists(root):
        raise ValueError(f"path {root} does not exist")
    if webp:
        monkaa_images = path.join(root, "monkaa__frames_cleanpass_webp")
        driving_images = path.join(root, "driving__frames_cleanpass_webp")
        flying_images = path.join(root, "flyingthings3d__frames_cleanpass_webp")
    else:
        monkaa_images = path.join(root, "monkaa__frames_cleanpass")
        driving_images = path.join(root, "driving__frames_cleanpass")
        flying_images = path.join(root, "flyingthings3d__frames_cleanpass")
    monkaa_disparity = path.join(root, "monkaa__disparity")
    driving_disparity = path.join(root, "driving__disparity")
    flying_disparity = path.join(root, "flyingthings3d__disparity")

    check_paths_exist(
        monkaa_images,
        monkaa_disparity,
        driving_images,
        driving_disparity,
        flying_images,
        flying_disparity,
    )

    driving, driving_test, _ = index_driving(
        driving_images,
        driving_disparity,
        webp,
        disparity_side,
        split=0.0,
    )
    flying, flying_test, _ = index_flyingthings(
        flying_images,
        flying_disparity,
        webp,
        disparity_side,
    )
    monkaa, monkaa_test, _ = index_monkaa(
        monkaa_images,
        monkaa_disparity,
        webp,
        disparity_side,
        split=0.0,
    )

    trainset = driving + flying + monkaa
    testset = driving_test + flying_test + monkaa_test

    return trainset, testset, read_pfm
