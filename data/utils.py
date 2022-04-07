from os import path, listdir
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch
from dataclasses import dataclass

IMAGENET_NORMALIZATION_PARAMS = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
normalize = transforms.Normalize(*IMAGENET_NORMALIZATION_PARAMS)


def check_paths_exist(*args):
    for a in args:
        if not a:
            raise ValueError("Specified path is None")
        if not path.exists(a):
            raise ValueError(f"path {a} does not exist")


@dataclass
class pad_parameters:
    h_pad: int
    w_pad: int
    heigh: int
    width: int


def pad_image(input: torch.Tensor):
    *_, h, w = input.shape
    h_pad = 16 - h % 16
    w_pad = 16 - w % 16
    # left, top, right, bottom
    res = TF.pad(input, (0, h_pad, w_pad, 0))
    return res, pad_parameters(h_pad, w_pad, h, w)


def pad_image_reverse(input: torch.Tensor, params: pad_parameters):
    p = params
    # top, left, heigh, width
    return TF.crop(input, p.h_pad, p.w_pad, p.heigh, p.width)


def match_images_disparities(left_folder, right_folder, disp_folder, input_extension):
    """
    Matches image pairs with disparity files into tuples of three
    """
    check_paths_exist(left_folder, right_folder, disp_folder)
    triplets = []
    for disparity_file in listdir(disp_folder):
        filename = path.splitext(disparity_file)[0]
        image_name = f"{filename}.{input_extension}"

        disp_file = path.join(disp_folder, disparity_file)
        left_image = path.join(left_folder, image_name)
        right_image = path.join(right_folder, image_name)
        if (
            path.isfile(disp_file)
            and path.isfile(left_image)
            and path.isfile(right_image)
        ):
            triplets.append((left_image, right_image, disp_file))
    return triplets
