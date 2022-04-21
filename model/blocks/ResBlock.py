from torch import nn


class BasicBlock(nn.Module):
    """
        Singular block which is a part of ResBlock
    """
    def __init__(
        self, in_channels, out_channels, kernel=3, stride=1, padding=1, dilation=1,norm_groups=4
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel, stride, padding, dilation, bias=False
            ),
            nn.GroupNorm(norm_groups,out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels, kernel, 1, padding, dilation, bias=False
            ),
            nn.GroupNorm(norm_groups,out_channels),
        )

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.GroupNorm(norm_groups,out_channels),
            )

    def forward(self, input):
        out = self.block(input)

        skip = input
        if self.downsample != None:
            skip = self.downsample(input)

        out += skip
        return out


class ResBlock(nn.Module):
    def __init__(
        self, blocks, in_channels, out_channels, stride, padding=1, dilation=1,norm_groups=4
    ):
        super().__init__()
        layers = []
        cur_channels = in_channels

        for i in range(blocks):
            layer = BasicBlock(
                cur_channels,
                out_channels,
                3,
                stride,
                padding,
                dilation,
                norm_groups,
            )
            layers.append(layer)
            stride = 1
            cur_channels = out_channels

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(input)
