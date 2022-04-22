import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torchvision import transforms
from typing import List,Tuple
from .utils import IMAGENET_NORMALIZATION_PARAMS, crop_image_to_multiple, imagenet_normalization
from .file_handling import read_file

__to_tensor = transforms.ToTensor()


class DisparityDataset(Dataset):
    def __init__(
        self,
        paths: List[Tuple[str, str, str]],
        random_crop=True,
        crop_shape=(256, 512),
        return_paths: bool = False,
        crop_to_multiple: bool = False,
        multiple: int = 16,
    ):
        self.image_paths = paths[:]
        self.random_crop = random_crop
        self.return_paths = return_paths
        self.crop_shape = None
        if random_crop:
            self.crop_shape = crop_shape
        self.__to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(*IMAGENET_NORMALIZATION_PARAMS)
        self.crop_to_multiple = crop_to_multiple
        self.multiple = multiple

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        left, right, disp = self.image_paths[index]
        left, right, disp = read_and_prepare(
            left, right, disp, self.random_crop, self.crop_shape, True
        )

        if self.crop_to_multiple:
            left = crop_image_to_multiple(left,multiple=self.multiple)
            right = crop_image_to_multiple(right,multiple=self.multiple)
            disp = crop_image_to_multiple(disp,multiple=self.multiple)

        if self.return_paths:
            return left, right, disp, self.image_paths[index]
        return left, right, disp

    def to_tensor(self, input):
        if not isinstance(input, torch.Tensor):
            return self.__to_tensor(input)
        return input


def read_and_prepare(
    left: str,
    right: str,
    disparity: str,
    random_crop=False,
    crop_shape=(256, 512),
    normalize=True,
    add_dim=False,
):
    left_data = read_file(left)
    right_data = read_file(right)

    if random_crop:
        i, j, w, h = transforms.RandomCrop.get_params(left_data, crop_shape)
        left_data = TF.crop(left_data, i, j, w, h)
        right_data = TF.crop(right_data, i, j, w, h)

    left_data = __to_tensor(left_data)
    right_data = __to_tensor(right_data)

    if normalize:
        left_data = imagenet_normalization(left_data)
        right_data = imagenet_normalization(right_data)

    if add_dim:
        left_data.unsqueeze_(0)
        right_data.unsqueeze_(0)

    if not disparity:
        return left_data, right_data, None

    disp_data = read_file(disparity, disparity=True)

    if random_crop:
        disp_data = TF.crop(disp_data, i, j, w, h)

    if not isinstance(disp_data, torch.Tensor):
        disp_data = __to_tensor(disp_data).squeeze(0)
        disp_data = disp_data.float() / 256

    if add_dim:
        disp_data.unsqueeze_(0)
    return left_data, right_data, disp_data


def assert_correct_shape(input: torch.Tensor):
    *_, h, w = input.shape
    if h < 256:
        raise AssertionError("height of an image below 256 pixels")
    if w < 256:
        raise AssertionError("width of an image below 256 pixels")
    if h % 16 != 0:
        raise AssertionError("height of an image not divisible by 4")
    if w % 16 != 0:
        raise AssertionError("width of an image not divisible by 4")
