import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torchvision import transforms

from .utils import IMAGENET_NORMALIZATION_PARAMS
from .file_handling import read_file


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
