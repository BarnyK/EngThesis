import time
from torch import from_numpy, le
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from os import path
from .indexes import *
from .file_handling import read_file
from tqdm import tqdm

IMAGENET_NORMALIZATION_PARAMS = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
normalize = transforms.Normalize(*IMAGENET_NORMALIZATION_PARAMS)


class DisparityDataset(Dataset):
    def __init__(
        self, paths: list[tuple[str, str, str]], random_crop=True, crop_shape=(256, 512)
    ):
        self.image_paths = paths[:]
        self.random_crop = random_crop
        if random_crop:
            self.crop_shape = crop_shape
        self.__to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(*IMAGENET_NORMALIZATION_PARAMS)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        left, right, disp = self.image_paths[index]
        left = read_file(left)
        right = read_file(right)
        disp = read_file(disp, disparity=True)

        ## Crop
        if self.random_crop:
            i, j, w, h = transforms.RandomCrop.get_params(left, (256, 512))
            left = TF.crop(left, i, j, w, h)
            right = TF.crop(right, i, j, w, h)
            disp = TF.crop(disp, i, j, w, h)

        left = self.to_tensor(left)
        right = self.to_tensor(right)
        disp = self.to_tensor(disp)

        left = self.normalize(left)
        right = self.normalize(right)

        if disp.dim() == 3:
            disp.squeeze_(0)
            disp = disp.float() / 256

        return left, right, disp

    def to_tensor(self, input):
        if not isinstance(input, torch.Tensor):
            return self.__to_tensor(input)
        return input


def pad_image(input):
    *_, h, w = input.shape
    h_pad = 16 - h % 16
    w_pad = 16 - w % 16
    res = TF.pad(input, (0, h_pad, w_pad, 0))
    return res, input.shape


def pad_image_reverse(input: torch.Tensor, original_shape):
    return TF.crop(input, *original_shape)
