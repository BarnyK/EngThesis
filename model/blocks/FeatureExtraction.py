from torch import nn

from ..utils import conv2d_norm_relu
from .ResBlock import ResBlock
from .SDEA import SDEABlock
from .SPPblock import SPPBlock


class FeatureExtraction(nn.Module):
    def __init__(self, max_disp:int=192, no_sdea:bool=False):
        """
            Initialize FeatureExtraction module.
            @max_disp: maximal disparity needed for SDEA block calculations
            @no_sdea: controls whether SDEA blocks or ResBlocks are used for 3rd and 4th layer
        """
        super().__init__()
        self.no_sdea = no_sdea
        self.initial = nn.Sequential(
            conv2d_norm_relu(3, 32, 3, 2, 1),
            conv2d_norm_relu(32, 32, 3, 1, 1),
            conv2d_norm_relu(32, 32, 3, 1, 1),
        )
        self.resblock1 = ResBlock(3, 32, 32, 1,norm_groups=4)
        self.resblock2 = ResBlock(16, 32, 64, 2,norm_groups=8)
        if self.no_sdea:
            self.resblock3 = ResBlock(3, 64, 128, 1, padding=1, dilation=1,norm_groups=8)
            self.resblock4 = ResBlock(3, 128, 128, 1, padding=2, dilation=2,norm_groups=8)
        else:
            self.sdea0_0 = SDEABlock(64, 128, max_disp // 4)
            self.sdea0_1 = SDEABlock(128, 128, max_disp // 4)
            self.sdea0_2 = SDEABlock(128, 128, max_disp // 4)
            self.sdea1_0 = SDEABlock(128, 128, max_disp // 4)
            self.sdea1_1 = SDEABlock(128, 128, max_disp // 4)
            self.sdea1_2 = SDEABlock(128, 128, max_disp // 4)

        self.spp = SPPBlock(128, 32, 64)

    def forward(self, left, right):
        left = self.initial(left)
        left = self.resblock1(left)
        left = self.resblock2(left)

        right = self.initial(right)
        right = self.resblock1(right)
        right = self.resblock2(right)

        left_skip, right_skip = left, right  # Skip from 2nd res block

        if self.no_sdea:
            left = self.resblock3(left)
            left = self.resblock4(left)

            right = self.resblock3(right)
            right = self.resblock4(right)

        else:
            left, right = self.sdea0_0(left, right)
            left, right = self.sdea0_1(left, right)
            left, right = self.sdea0_2(left, right)
            left, right = self.sdea1_0(left, right)
            left, right = self.sdea1_1(left, right)
            left, right = self.sdea1_2(left, right)

        left = self.spp(left, left_skip)
        right = self.spp(right, right_skip)

        return left, right
