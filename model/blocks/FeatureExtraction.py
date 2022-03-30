import torch
import torch.nn.functional as F
from torch import nn

from ..utils import conv2d_norm_relu
from .ResBlock import ResBlock


class InitialFeatureExtraction(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_0 = conv2d_norm_relu(3, 32, 3, 2, 1)
        self.conv_1 = conv2d_norm_relu(32, 32, 3, 1, 1)
        self.conv_2 = conv2d_norm_relu(32, 32, 3, 1, 1)
        self.resblock1 = ResBlock(3, 32, 32, 1)
        self.resblock2 = ResBlock(16, 32, 64, 2)

    def _forward(self, input):
        out = self.conv_0(input)
        out = self.conv_1(out)
        out = self.conv_2(out)
        return out

    def forward(self, input):
        out = self.conv_0(input)
        out = self.conv_1(out)
        out = self.conv_2(out)
        out = self.resblock1(out)
        out = self.resblock2(out)
        return out
