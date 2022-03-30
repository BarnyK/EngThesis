import torch
import torch.nn.functional as F
from torch import nn


class SDEABlock(nn.Module):
    def __init__(self, in_channels: int, out_channels:int, maxdisp):
        super().__init__()
        self.maxdisp = maxdisp
        self.g1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.g2 = nn.Conv2d(out_channels, 1, 1, 1, 0)

        self.channel_scaling = None
        if in_channels != out_channels:
            self.channel_scaling = nn.Conv2d(in_channels,out_channels,1,1)

    def forward(self, left, right):
        # two convolution layers G1
        left_g1 = self.g1(left)
        right_g1 = self.g1(right)

        # scaling the input channels
        if self.channel_scaling:
            left = self.channel_scaling(left)
            right = self.channel_scaling(right)

        # 1x1 convolution layer G2
        left_g2 = self.g2(left_g1)
        right_g2 = self.g2(right_g1)

        # Weight calculation
        left_weight = torch.sigmoid(self.weight_matrix_calculation(left_g2, right_g2))
        right_weight = torch.sigmoid(self.weight_matrix_calculation(right_g2, left_g2))

        # Elementwise multiplication
        left_out = left_g1 * left_weight
        right_out = right_g1 * right_weight

        # Elementwise sum
        left_out = left_out + left
        right_out = right_out + right

        return left_out, right_out

    def weight_matrix_calculation(self, left, right):
        """
        Then we are on one G2, for each point x on G2,
        we find the point x with the minimum difference
        on the other G2 in themax-disp range
        """
       
        weight_volume = torch.zeros_like(left)
        
        for j in range(left.shape[3]):
            left_bound = max(0, j - self.maxdisp)
            right_bound = min(left.shape[3], j + self.maxdisp)
            diff = right[:, 0, :, left_bound:right_bound] - left[:, 0, :, j:j+1]
            diff = torch.abs(diff)
            v, _ = diff.min(2)
            weight_volume[:, 0, :, j] = v
        return weight_volume



class Test(nn.Module):
    def __init__(self,in_layers,maxdisp):
        super().__init__()
        self.l1 = SDEABlock(in_layers,in_layers,maxdisp)
        self.l2 = SDEABlock(in_layers,in_layers,maxdisp)
    def forward(self,x,y):
        return self.l2(*self.l1(x,y))

if __name__ == "__main__":
    # torch.backends.cudnn.benchmark = True
    shape = (1,32,256//4,512//4)
    left = torch.rand(shape).requires_grad_().cuda()
    right = torch.rand(shape).requires_grad_().cuda()
    layer = Test(shape[1],48).cuda()
    import time
    times = []
    for _ in range(5):
        left = torch.rand(shape).requires_grad_().cuda()
        right = torch.rand(shape).requires_grad_().cuda()
        st = time.time()
        
        out1,out2 = layer(left,right)
        tt = time.time()
        # print(tt - st)
        times.append(tt - st)
    kap1 = (left - out1).sum()
    kap2 = (right - out2).sum()
    time.sleep(10)
    kap1.backward()
    # kap2.backward()
    print(kap1)
    print(kap2)
    print(sum(times[1:])/(len(times)-1))
    import matplotlib.pyplot as plt

    plt.plot(times[1:])
    plt.show()

