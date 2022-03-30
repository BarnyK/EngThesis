from torch import nn

def conv2d_norm(in_, out, kernel, stride, padding=0, dilation=1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_, out, kernel, stride, padding, dilation, bias=False),
        nn.BatchNorm2d(out),
    )


def conv2d_norm_relu(in_, out, kernel, stride, padding=0, dilation=1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_, out, kernel, stride, padding, dilation, bias=False),
        nn.BatchNorm2d(out),
        nn.ReLU(inplace=True),
    )


def conv3d_norm(in_, out, kernel, stride, padding=0, dilation=1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_, out, kernel, stride, padding, dilation, bias=False),
        nn.BatchNorm3d(out),
    )


def conv3d_norm_relu(in_, out, kernel, stride, padding=0, dilation=1):
    return nn.Sequential(
        nn.Conv3d(in_, out, kernel, stride, padding, dilation, bias=False),
        nn.BatchNorm3d(out),
        nn.ReLU(inplace=True),
    )