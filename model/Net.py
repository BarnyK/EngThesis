import time
import torch
from torch.nn import functional as F
from torch import nn
from .blocks import SDEABlock, SPPBlock, StackedHourglassModule, InitialFeatureExtraction


class Net(nn.Module):
    ## Main module for Network
    def __init__(self, max_disp=192):
        super().__init__()
        self.maxdisp = max_disp
        self.initial_extraction = InitialFeatureExtraction()
        self.sdea0 = SDEABlock(64, 128, max_disp)
        self.sdea1 = SDEABlock(128, 128, max_disp)
        self.spp = SPPBlock(128, 32, 64)
        self.stacked_hourglass = StackedHourglassModule()

        self.disparities = torch.arange(max_disp, requires_grad=False).reshape((1, max_disp, 1, 1))

    def to(self,device):
        new_self = super().to(device)
        new_self.disparities = new_self.disparities.to(device)

        return new_self

    def forward(self, left, right):
        left = self.initial_extraction(left)
        right = self.initial_extraction(right)
        
        left_skip, right_skip = left, right
        # 64 x H/4 x W/4
        left, right = self.sdea0(left, right)

        # 128 x H/4 x W/4
        left, right = self.sdea1(left, right)

        # 128 x H/4 x W/4
        left = self.spp(left, left_skip)
        right = self.spp(right, right_skip)

        # 32 x H/4 x W/4
        cost = self.create_cost_volume(left, right)

        # 64 x maxdisp / 4 x H/4 x W/4
        out1, out2, out3 = self.stacked_hourglass(cost)

        # 1 x maxdisp / 4 x H/4 x W/4
        
        out3 = self.upsample_regression(out3)
        # H x W
        if not self.training:
            return out3

        out1 = self.upsample_regression(out1)
        out2 = self.upsample_regression(out2)
        return out1, out2, out3

    def upsample_regression(self, input):
        # Upsample
        out = F.interpolate(input, scale_factor=4, mode="trilinear", align_corners=True)
        # Remove the channels dimension, whch is 1 at this point
        out = torch.squeeze(out, 1)
        # Softmax over dimension with disparity
        out = F.softmax(out, 1)
        # Multiply weight by range of disparities used
        out = out * self.disparities
        # Sum to get the final disparity
        out = torch.sum(out, 1)
        return out


    def create_cost_volume(self, left: torch.Tensor, right: torch.Tensor):
        # Initialize volume with zeros on the same device as input
        cost = torch.empty(
            (
                left.shape[0],
                left.shape[1] + right.shape[1],
                self.maxdisp // 4,
                left.shape[2],
                left.shape[3],
            ),
            device=left.device,
        )
        ch = left.shape[1]
        # Copy from feature matrices to cost volume
        cost[:, :ch, 0, :, :] = left
        cost[:, ch:, 0, :, :] = right
        for i in range(1, self.maxdisp // 4):
            cost[:, :ch, i, :, i:] = left[:, :, :, i:]
            cost[:, ch:, i, :, i:] = right[:, :, :, :-i]
        return cost



def main():
    net = Net(192).to("cuda")
    left = torch.rand((1, 3, 256, 512)).to("cuda")
    right = torch.rand((1, 3, 256, 512)).to("cuda")
    net.train(True)
    with torch.cuda.amp.autocast():
        xd = net(left, right)
    if type(xd) == tuple:
        print(xd[0].shape, xd[1].shape, xd[2].shape)
    else:
        print(xd.shape)

if __name__ == "__main__2":
    from blocks.ResBlock import BaseBlock, ResBlock
    x = BaseBlock(32,32,3,1)
    y = BaseBlock(32,64,3,2)
    left = torch.rand((1, 32, 256, 512))
    k = x(left)
    k = y(k)

    resblock = ResBlock(3,32,64,2)
    print(resblock)
    out = resblock(left)
    print(out.shape)

if __name__ == "__main__":
    main()