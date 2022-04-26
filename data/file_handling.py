import re
from os import path

import numpy as np
import torch
from PIL import Image

from .utils import to_tensor

def __read_PFM(file: str):
    """
    readPFM reads .pfm files and returns them as
    Written by authors of SceneFlow datasets
    """
    file = open(file, "rb")

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == "PF":
        color = True
    elif header.decode("ascii") == "Pf":
        color = False
    else:
        raise Exception("Not a PFM file.")

    dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception("Malformed PFM header.")

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0:  # little-endian
        endian = "<"
        scale = -scale
    else:
        endian = ">"  # big-endian

    data = np.fromfile(file, endian + "f")
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def read_pfm(filename: str):
    data, _ = __read_PFM(filename)
    return torch.from_numpy(data.copy())


def read_image(filename: str) -> Image:
    img = Image.open(filename)
    return img.convert("RGB")


def read_uint16png(filename):
    img = Image.open(filename)
    img = to_tensor(img).squeeze(0)
    img = img.float() / 256
    return img

def read_file(filename, disparity=False):
    _, ext = path.splitext(filename)
    ext = ext.lower()
    if ext == ".pfm":
        return read_pfm(filename)
    elif disparity:
        return read_uint16png(filename)
    else:
        return read_image(filename)