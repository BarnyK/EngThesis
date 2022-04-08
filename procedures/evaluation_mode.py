import os
import time
from os import path
from pickle import UnpicklingError

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from data.dataset import DisparityDataset, assert_correct_shape, read_and_prepare
from data.file_handling import read_file
from data.indexing import index_set
from data.utils import (
    check_paths_exist,
    imagenet_normalization,
    pad_image,
    pad_image_reverse,
    pad_image_,
    pad_image_reverse_,
    pad_parameters,
)
from measures import error_3p, error_epe
from model import Net
from model.utils import choose_device, load_model
from PIL import Image
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torchvision import transforms

from procedures.train_mode import prepare_model_optim_scaler


def evaluate_one(
    left_image: str,
    right_image: str,
    result_image: str,
    disparity_image: str,
    max_disp: int,
    load_file: str,
    cpu: bool,
    no_sdea: bool,
    **kwargs,
):
    try:
        if disparity_image:
            check_paths_exist(left_image, right_image, disparity_image)
        else:
            check_paths_exist(left_image, right_image)
    except ValueError as er:
        print(er)
        return

    if max_disp <= 0 or max_disp % 4 != 0:
        print("max_disp must be integer bigger than 0 divisible by 4")
        return

    try:
        device = choose_device(cpu)
    except Exception as ex:
        print(ex)
        return

    try:
        left, right, gt = read_and_prepare(
            left_image, right_image, disparity_image, add_dim=True
        )

    except Exception as ex:
        print(ex)
        return

    left, s = pad_image_(left)
    right, _ = pad_image_(right)

    assert_correct_shape(left)
    assert_correct_shape(right)

    net = Net(max_disp, no_sdea)
    try:
        if load_file:
            state, *_ = load_model(load_file)
            if state:
                net.load_state_dict(state)
            else:
                print("Couldn't load model from given file")
                return
    except FileNotFoundError:
        print("Given load file doesn't exist")
        return

    net.to(device)
    net.eval()
    with torch.inference_mode():
        prediction = net(left, right)

    pred = pad_image_reverse_(prediction, s)

    if gt is not None:
        mask = torch.logical_and(gt < max_disp, gt > 0)
        loss = F.smooth_l1_loss(gt[mask],pred[mask])
        epe = error_epe(gt[mask], pred[mask])
        e3p = error_3p(gt[mask], pred[mask])
        print("Loss: ", loss.item())
        print("Endpoint error: ", epe)
        print("3 pixel error: ", e3p)

    if result_image:
        pred = pred.squeeze(0)
        res = np.array(pred.cpu(), dtype=np.uint8)
        prediction_image = Image.fromarray(res)
        os.makedirs(os.path.dirname(result_image), exist_ok=True)
        if not result_image.lower().endswith(".png"):
            result_image += ".png"
        prediction_image.save(result_image)
        print(f"Saved file to {result_image}")


def eval_dataset(
    dataset_name, max_disp, cpu, no_sdea, load_file, log_file, only_testset, **kwargs
):
    try:
        device = choose_device(cpu)
    except Exception as ex:
        print(ex)
        return

    if max_disp <= 0 or max_disp % 4 != 0:
        print("max_disp must be integer bigger than 0 divisible by 4")
        return

    try:
        trainset, testset = index_set(dataset_name, **kwargs)
        trainset = DisparityDataset(trainset, random_crop=False, return_paths=True)
        testset = DisparityDataset(testset, random_crop=False, return_paths=True)
        trainloader = DataLoader(
            trainset,
            1,
            shuffle=False,
            num_workers=2,
            pin_memory=not cpu,
        )
        testloader = DataLoader(
            testset, 1, shuffle=False, num_workers=2, pin_memory=not cpu
        )
    except ValueError as er:
        print(er)
        return

    try:
        net, *_ = prepare_model_optim_scaler(
            load_file, device, max_disp, no_sdea, 0
        )
    except FileNotFoundError as err:
        print("Could not find given load_file ", load_file)
        return
    except UnpicklingError as err:
        print("Could not load given file, because it has wrong format ", load_file)
        return
    except RuntimeError as err:
        print(err)
        return

    if log_file and os.path.isfile(log_file):
        print("Given log file already exists")
        return

    def save_log(mode, path_to_left_img, t, loss, epe, e3p):
        """
        Function for appending metrics to log file
        """
        if log_file:
            log = f"{mode},{path_to_left_img},{t},{loss},{epe},{e3p}\n"
            with open(log_file, "a") as f:
                f.write(log)

    def eval_on_loader(loader, mode):
        for i, (left, right, gt, paths) in tqdm(enumerate(loader),total=len(loader)):

            left, pad_params = pad_image_(left)
            right, _ = pad_image_(right)

            left = left.to(device)
            right = right.to(device)
            mask = torch.logical_and(gt < max_disp, gt > 0)
            gt = gt.to(device, non_blocking=True)

            with torch.inference_mode(), autocast(enabled=device.type == "cuda"):
                st = time.time()
                prediction = net(left, right)
                et = time.time()
            prediction = pad_image_reverse_(prediction, pad_params)

            time_taken = et - st
            loss = F.smooth_l1_loss(gt[mask], prediction[mask]).item()
            epe = error_epe(gt, prediction, max_disp)
            e3p = error_3p(gt, prediction, max_disp)
            # print(i, paths[0][0])
            # print("Time taken:", time_taken)
            # print("Loss: ", loss)
            # print("Endpoint error:", epe)
            # print("3 pixel error:", e3p)
            save_log(mode, paths[0][0], time_taken, loss, epe, e3p)

    net.eval()
    if not only_testset:
        print("Trainset")
        eval_on_loader(trainloader, "train")
    print("Test set")
    eval_on_loader(testloader, "test")
