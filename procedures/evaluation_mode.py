import os
import time
from pickle import UnpicklingError

import numpy as np
import torch
import torch.nn.functional as F
from data.dataset import DisparityDataset, assert_correct_shape, read_and_prepare
from data.indexing import index_set
from data.utils import check_paths_exist, pad_image_reverse, pad_image_to_multiple
from measures import error_3p, error_epe
from model import Net
from model.utils import choose_device, load_model
from PIL import Image
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from procedures.train_mode import prepare_model_optim_scaler

from measures.logging import Metrics


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

    except FileNotFoundError as ex:
        print(ex)
        return

    left, p = pad_image_to_multiple(left)
    left = left.to(device, non_blocking=True)
    right, _ = pad_image_to_multiple(right)
    right = right.to(device, non_blocking=True)
    if gt is not None:
        gt = gt.to(device, non_blocking=True)

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

    pred = pad_image_reverse(prediction, p)

    if gt is not None:
        mask = torch.logical_and(gt < max_disp, gt > 0)
        loss = F.smooth_l1_loss(gt[mask], pred[mask])
        epe = error_epe(gt[mask], pred[mask])
        e3p = error_3p(gt[mask], pred[mask])
        print("Loss: ", loss.item())
        print("Endpoint error: ", epe)
        print("3 pixel error: ", e3p)

    if result_image:
        pred = pred.squeeze(0)
        res = np.array(pred.cpu()*256, dtype=np.uint16)
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

    if max_disp <= 0 or max_disp % 16 != 0:
        print("max_disp must be integer bigger than 0 divisible by 16")
        return

    try:
        trainset, testset, disp_func = index_set(dataset_name, **kwargs)
        trainset = DisparityDataset(
            trainset, disp_func, random_crop=False, return_paths=True, crop_to_multiple=True
        )
        testset = DisparityDataset(
            testset, disp_func, random_crop=False, return_paths=True, crop_to_multiple=True
        )
        trainloader = DataLoader(
            trainset,
            1,
            shuffle=False,
            num_workers=4,
            pin_memory=not cpu,
        )
        testloader = DataLoader(
            testset, 1, shuffle=False, num_workers=2, pin_memory=not cpu
        )
    except ValueError as er:
        print(er)
        return

    try:
        net, *_ = prepare_model_optim_scaler(load_file, device, max_disp, no_sdea, 0)
    except FileNotFoundError as err:
        print("Could not find given load_file ", load_file, err)
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
        m = Metrics()
        for _, (left, right, gt, paths) in tqdm(enumerate(loader), total=len(loader)):
            left = left.to(device)
            right = right.to(device)
            mask = torch.logical_and(gt < max_disp, gt > 0)
            gt = gt.to(device, non_blocking=True)

            with torch.inference_mode(), autocast(enabled=device.type == "cuda"):
                st = time.time()
                prediction = net(left, right)
                et = time.time()

            time_taken = et - st
            loss = F.smooth_l1_loss(gt[mask], prediction[mask]).item()
            epe = error_epe(gt[mask], prediction[mask])
            e3p = error_3p(gt[mask], prediction[mask])
            m.add(loss, epe, e3p, 1, 1)
            save_log(mode, paths[0][0], time_taken, loss, epe, e3p)
        m.end()
        print(m)

    net.eval()
    if not only_testset:
        print("Trainset")
        eval_on_loader(trainloader, "train")
    print("Test set")
    eval_on_loader(testloader, "test")
