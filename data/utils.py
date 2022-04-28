from dataclasses import dataclass
from math import ceil
from os import listdir, path

import torch
import torchvision.transforms.functional as TF
from torchvision import transforms

IMAGENET_NORMALIZATION_PARAMS = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
imagenet_normalization = transforms.Normalize(*IMAGENET_NORMALIZATION_PARAMS)
to_tensor = transforms.ToTensor()


def check_paths_exist(*args):
    for a in args:
        if not a:
            raise ValueError("Specified path is None")
        if not path.exists(a):
            raise ValueError(f"path {a} does not exist")


@dataclass
class CropParams:
    top: int = 0
    left: int = 0
    height: int = 0
    width: int = 0


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


def pad_image_to_multiple(
    input: torch.Tensor, left: bool = True, top: bool = True, multiple: int = 16
):
    if multiple <= 0:
        raise ValueError("multiple should be bigger than 0")
    *_, h, w = input.shape
    h_pad = ceil(h / multiple) * multiple - h
    w_pad = ceil(w / multiple) * multiple - w
    if left and top:
        res = TF.pad(input, (w_pad, h_pad, 0, 0))
        return res, CropParams(h_pad, w_pad, h, w)
    elif left and not top:
        res = TF.pad(input, (w_pad, 0, 0, h_pad))
        return res, CropParams(0, w_pad, h, w)
    elif not left and top:
        res = TF.pad(input, (0, h_pad, w_pad, 0))
        return res, CropParams(h_pad, 0, h, w)
    else:
        res = TF.pad(input, (0, 0, w_pad, h_pad))
        return res, CropParams(0, 0, h, w)


def pad_image_reverse(input: torch.Tensor, params: CropParams):
    p = params
    return TF.crop(input, p.top, p.left, p.height, p.width)


def crop_image_to_multiple(
    input: torch.Tensor, left: bool = False, top: bool = False, multiple: int = 16
):
    if multiple <= 0:
        raise ValueError("multiple should be bigger than 0")
    *_, h, w = input.shape
    h_crop = h - (h // multiple) * multiple
    w_crop = w - (w // multiple) * multiple
    if left and top:
        res = TF.crop(input, 0, 0, h - h_crop, w - w_crop)
        return res
    elif left and not top:
        res = TF.crop(input, h_crop, 0, h - h_crop, w - w_crop)
        return res
    elif not left and top:
        res = TF.crop(input, 0, w_crop, h - h_crop, w - w_crop)
        return res
    else:
        res = TF.crop(input, h_crop, w_crop, h - h_crop, w - w_crop)
        return res


from torch.nn.functional import interpolate


def interpolate_image(input: torch.Tensor):

    if input.dim() == 3:
        i = input.unsqueeze(1)
    else:
        i = input

    n, c, h, w = i.shape
    h_ = ceil(h / 16) * 16
    w_ = ceil(w / 16) * 16
    i = interpolate(i, (h_, w_), mode="bicubic", align_corners=False)
    if input.dim() == 3:
        i.squeeze_(1)
    return i, (h, w)


def interpolate_image_reverse(input, og_shape):
    if input.dim() == 3:
        i = input.unsqueeze(1)
    else:
        i = input

    n, c, h, w = i.shape
    i = interpolate(i, og_shape, mode="bicubic", align_corners=False)
    if input.dim() == 3:
        i.squeeze_(1)
    return i
