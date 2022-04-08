import torch
from torch import nn
from torch.optim import Optimizer

from . import Net


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


def choose_device(cpu: bool):
    if cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    raise Exception("Pytorch did not find cuda")


def save_model(model: Net, optimizer: Optimizer, scaler, savepath: str):
    res = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
    }
    torch.save(res, savepath)
    return res


def load_model(loadpath: str, device: torch.device = torch.device("cpu")):
    state = torch.load(loadpath, map_location=device)
    return state.get("model"), state.get("optimizer"), state.get("scaler")
