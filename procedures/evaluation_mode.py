import enum
from json import load
import os
from pickle import UnpicklingError
import time
from PIL import Image
import numpy as np
from torch.cuda.amp import autocast
import torch
from data.utils import check_paths_exist, pad_image, pad_image_reverse, normalize
from data.file_handling import read_file
from data.dataset import DisparityDataset
from data.indexing import index_set
from torch.utils.data import DataLoader
from measures import error_3p, error_epe
from model import Net
from model.utils import choose_device, load_model
from torchvision import transforms
import torch.nn.functional as F
from os import path
from procedures.train_mode import prepare_model_optim_scaler, Metrics


def evaluate(
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
    if max_disp is None or max_disp <= 0:
        raise ValueError("max_disp must be integer bigger than 0")

    check_paths_exist(left_image, right_image)

    try:
        device = choose_device(cpu)
    except Exception as ex:
        print(ex)
        return

    # Load and pre-process files
    left = read_file(left_image)
    right = read_file(right_image)
    to_tensor = transforms.ToTensor()

    left = to_tensor(left)
    right = to_tensor(right)

    left = normalize(left).unsqueeze(0).to(device)
    right = normalize(right).unsqueeze(0).to(device)

    if left.shape != right.shape:
        print("Images of different shapes can't be passed to the network")
        return
    left, s = pad_image(left)
    right, _ = pad_image(right)

    # Load model
    net = Net(max_disp, no_sdea)
    if load_file:
        state, *_ = load_model(load_file)
        if state:
            net.load_state_dict(state)
        else:
            print("Couldn't load model from given file")
            return

    net.to(device)
    net.eval()

    # Pass through the network
    with torch.inference_mode():
        _ = net.forward(left, right)
        st = time.time()
        disp = net.forward(left, right)
        et = time.time()
        print("Pass took: ", round(et - st, 2), "seconds")

    # Create stats
    disp = pad_image_reverse(disp, s)
    if disparity_image and os.path.exists(disparity_image):
        gt = read_file(disparity_image, disparity=True)
        if not isinstance(gt, torch.Tensor):
            gt = to_tensor(gt).float() / 256
        else:
            gt = gt.unsqueeze(0)
        gt = gt.to(device)
        if gt.shape == disp.shape:
            print("EPE:", error_epe(gt, disp,net.maxdisp))
            print("3p:", error_3p(gt, disp,net.maxdisp))
        else:
            print(
                "Can't create measures if output disparity is different shape than ground truth"
            )

    # Save image
    disp = disp.squeeze(0)
    res = np.array(disp.cpu(), dtype=np.uint8)
    disp_image = Image.fromarray(res)
    os.makedirs(os.path.dirname(result_image), exist_ok=True)
    if not result_image.lower().endswith(".png"):
        result_image += ".png"
    disp_image.save(result_image)
    print(f"Saved file to {result_image}")


def eval_dataset(dataset_name, max_disp, cpu, no_sdea, load_file, log_file, **kwargs):
    try:
        device = choose_device(cpu)
    except Exception as ex:
        print(ex)
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
            pin_memory=True,
        )
        testloader = DataLoader(
            testset, 1, shuffle=False, num_workers=2, pin_memory=True
        )
    except ValueError as er:
        print(er)
        return

    try:
        net, optimizer, scaler = prepare_model_optim_scaler(
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
        for i, (left, right, gt, paths) in enumerate(loader):
            print(i, paths[0][0])

            left, pad_params = pad_image(left)
            right, _ = pad_image(right)

            left = left.to(device)
            right = right.to(device)
            mask = gt > 0
            gt = gt.to(device, non_blocking=True)

            with torch.inference_mode(), autocast(enabled=device.type == "cuda"):
                st = time.time()
                prediction = net(left, right)
                et = time.time()
            prediction = pad_image_reverse(prediction, pad_params)

            time_taken = et - st
            loss = F.smooth_l1_loss(gt[mask], prediction[mask]).item()
            epe = error_epe(gt, prediction,max_disp)
            e3p = error_3p(gt, prediction,max_disp)
            print("Time taken:", time_taken)
            print("Loss: ", loss)
            print("Endpoint error:", epe)
            print("3 pixel error:", e3p)
            save_log(mode, paths[0][0], time_taken, loss, epe, e3p)

    net.eval()
    print("Trainset")
    eval_on_loader(trainloader, "train")
    print("Test set")
    eval_on_loader(testloader, "test")
