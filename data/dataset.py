import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torchvision import transforms

from .utils import IMAGENET_NORMALIZATION_PARAMS, imagenet_normalization
from .file_handling import read_file

__to_tensor = transforms.ToTensor()


class DisparityDataset(Dataset):
    def __init__(
        self,
        paths: list[tuple[str, str, str]],
        random_crop=True,
        crop_shape=(256, 512),
        return_paths: bool = False,
    ):
        self.image_paths = paths[:]
        self.random_crop = random_crop
        self.return_paths = return_paths
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

    left_data.unsqueeze_(0)
    right_data.unsqueeze_(0)

    if not disparity:
        return left_data, right_data, None

    disp_data = read_file(disparity, disparity=True)

    if random_crop:
        disp_data = TF.crop(disp_data, i, j, w, h)

    if not isinstance(disp_data, torch.Tensor):
        disp_data = __to_tensor(disp_data)
        disp_data = disp_data.float() / 256

    disp_data.unsqueeze_(0)
    return left_data, right_data, disp_data

def assert_correct_shape(input: torch.Tensor):
    *_,h,w = input.shape
    if h < 256:
        raise AssertionError("height of an image below 256 pixels")
    if w < 256:
        raise AssertionError("width of an image below 256 pixels")
    if h%16 != 0:
        raise AssertionError("height of an image not divisible by 4")
    if w%16 != 0:
        raise AssertionError("width of an image not divisible by 4")