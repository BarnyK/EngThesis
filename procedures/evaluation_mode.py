import os
import time
from importlib.resources import path

import torch
from data.dataset import check_paths_exist, pad_image, pad_image_reverse
from data.file_handling import read_file
from measures import error_3p, error_epe
from model import Net
from model.utils import choose_device
from torchvision import transforms


def evaluate(
    left_image: str,
    right_image: str,
    result_image: str,
    disparity_image: str,
    max_disp: int,
    load_file: str,
    cpu: bool,
    timed: bool,
    **kwargs
):
    if max_disp is None or max_disp <= 0:
        raise ValueError("max_disp must be integer bigger than 0")

    check_paths_exist(left_image, right_image)

    try:
        device = choose_device(cpu)
    except Exception as ex:
        print(ex)
        return

    left = read_file(left_image)
    right = read_file(right_image)
    to_tensor = transforms.ToTensor()
    left = to_tensor(left).unsqueeze(0).to(device)
    right = to_tensor(right).unsqueeze(0).to(device)
    if left.shape != right.shape:
        print("Images of different shapes can't be passed to the network")
        return
    left, s = pad_image(left)
    right, _ = pad_image(right)

    m = Net(max_disp)

    if load_file:
        state = torch.load(load_file)
        if "model" in state:
            m.load_state_dict(state["model"])
        else:
            print("Couldn't load given load_file")
            return

    m.to(device)
    m.eval()

    with torch.no_grad():
        _ = m.forward(left, right)
        st = time.time()
        disp = m.forward(left, right)
        et = time.time()
        print(et - st)

    disp = pad_image_reverse(disp, s)
    print(disp.shape)

    if disparity_image and os.path.exists(disparity_image):
        gt = read_file(disparity_image, disparity=True)
        gt = to_tensor(gt).to(device)
        if gt.shape == disp.shape:
            print("EPE:", error_epe(gt, disp))
            print("3p:", error_3p(gt, disp))
            print(gt.shape)
        else:
            print(
                "Can't create measures if output disparity is different shape than ground truth"
            )

    disp_image = transforms.ToPILImage("L")(disp)

    os.makedirs(os.path.dirname(result_image), exist_ok=True)
    if not result_image.lower().endswith(".png"):
        result_image += ".png"
    disp_image.save(result_image)
    print(f"Saved file to {result_image}")
