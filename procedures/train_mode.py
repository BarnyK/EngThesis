from dataclasses import dataclass
from os import path
from pickle import UnpicklingError
from time import time

import torch
import torch.nn.functional as F
from data import DisparityDataset, index_set
from data.utils import pad_image, pad_image_reverse
from measures import error_3p, error_epe
from model import Net
from model.utils import choose_device, load_model, save_model
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(
    epochs: int,
    batch_size: int,
    learning_rate: float,
    dataset_name: str,
    cpu: bool,
    max_disp: int,
    load_file: str,
    save_file: str,
    log_file: str,
    eval_each_epoch: int,
    no_sdea: bool,
    iters_to_accumulate: int,
    **kwargs,
):
    torch.manual_seed(1111)

    if save_file is not None:
        save_filename = path.basename(save_file)
        if log_file and path.isdir(log_file):
            log_file = path.join(log_file, f"{save_filename}.log")

    def save_log(epoch, train_metrics: Metrics, test_metrics: Metrics):
        """
        Function for appending metrics to log file
        """
        trm = train_metrics
        tm = test_metrics
        if log_file:
            log = f"{save_filename},{dataset_name},{learning_rate},{epoch},"
            log += f"{trm.time_taken},{trm.running_loss},{trm.epe},{trm.e3p},"
            log += f"{tm.time_taken},{tm.running_loss},{tm.epe},{tm.e3p}\n"
            with open(log_file, "a") as f:
                f.write(log)

    try:
        device = choose_device(cpu)
    except Exception as ex:
        print(ex)
        return

    # Create dataloaders
    try:
        trainset, testset = index_set(dataset_name, **kwargs)
        trainset = DisparityDataset(trainset)
        testset = DisparityDataset(testset, random_crop=False)
        trainloader = DataLoader(
            trainset,
            batch_size,
            shuffle=True,
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
            load_file, device, max_disp, no_sdea, learning_rate
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

    try:
        for epoch in range(epochs):
            print(f"Epoch {epoch+1} out of {epochs}")
            print("Training loop")

            trm = training_loop(
                net,
                trainloader,
                trainset,
                optimizer,
                scaler,
                device,
                iters_to_accumulate,
            )
            if save_file:
                save_model(net, optimizer, scaler, f"{save_file}-{epoch}.tmp")

            print("Training metrics:")
            print(trm)

            tm = Metrics()
            if eval_each_epoch > 0 and (epoch + 1) % eval_each_epoch == 0:
                print("Testing loop")
                tm = testing_loop(net, testloader, testset, device)
                print("Test metrics:")
                print(tm)

            save_log(epoch, trm, tm)
    except KeyboardInterrupt:
        print("Received KeyboardInterrupt, closing")
        return

    if save_file:
        save_model(net, optimizer, scaler, save_file)


def prepare_model_optim_scaler(
    load_file: str,
    device: torch.device,
    max_disp: int,
    no_sdea: bool,
    learning_rate: float,
):
    model_state, optim_state, scaler_state = None, None, None
    if load_file:
        model_state, optim_state, scaler_state = load_model(load_file)

    net = Net(max_disp, no_sdea).to(device)
    if model_state:
        net.load_state_dict(model_state)

    optimizer = torch.optim.Adam(net.parameters(), learning_rate, eps=1e-6)
    if optim_state:
        optimizer.load_state_dict(optim_state)
        optimizer.param_groups[0]["lr"] = learning_rate

    scaler = GradScaler(init_scale=256, enabled=device.type == "cuda")
    if scaler_state:
        scaler.load_state_dict(scaler_state)

    return net, optimizer, scaler


def training_loop(
    net: Net,
    trainloader: DataLoader,
    trainset: DisparityDataset,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    iters_to_accumulate: int = 5,
):
    net.train()
    st = time()
    rl = 0.0
    epe_sum = 0
    e3p_sum = 0
    for i, (left, right, gt) in tqdm(enumerate(trainloader), total=len(trainloader)):
        left = left.to(device, non_blocking=True)
        right = right.to(device, non_blocking=True)
        mask = gt > 0
        gt = gt.to(device, non_blocking=True)

        with autocast(enabled=device.type == "cuda"):
            d1, d2, d3 = net(left, right)
            loss = (
                0.5 * F.smooth_l1_loss(d1[mask], gt[mask])
                + 0.7 * F.smooth_l1_loss(d2[mask], gt[mask])
                + F.smooth_l1_loss(d3[mask], gt[mask])
            )
        scaler.scale(loss / iters_to_accumulate).backward()
        if (i + 1) % iters_to_accumulate == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(True)

        # Metrics
        rl += loss.item() * d3.shape[0]
        epe_sum += error_epe(gt, d3) * d3.shape[0]
        e3p_sum += error_3p(gt, d3) * d3.shape[0]

    train_time = time() - st
    rl_avg = rl / len(trainset)
    epe_avg = epe_sum / len(trainset)
    e3p_avg = e3p_sum / len(trainset)
    return Metrics(train_time, rl_avg, epe_avg, e3p_avg)


def testing_loop(
    net: Net, testloader: DataLoader, testset: DisparityDataset, device: torch.device
):
    net.eval()
    st = time()
    rl = 0.0
    epe_sum = 0.0
    e3p_sum = 0.0
    print("Evaluating on test set")
    for i, (left, right, gt) in tqdm(enumerate(testloader), total=len(testloader)):
        left = left.to(device)
        right = right.to(device)
        mask = gt != 0
        gt = gt.to(device)
        left, og = pad_image(left)
        right, og = pad_image(right)
        with torch.inference_mode():
            with autocast(device.type == "cuda"):
                d = net(left, right)
                d = pad_image_reverse(d, og)
                epe = error_epe(gt, d) * d.shape[0]
                e3p = error_3p(gt, d) * d.shape[0]
                loss = F.smooth_l1_loss(d[mask], gt[mask])
            # metrics
            rl += loss * d.shape[0]
            epe_sum += epe * d.shape[0]
            e3p_sum += e3p * d.shape[0]

    train_time = time() - st
    rl_avg = rl / len(testset)
    epe_avg = epe_sum / len(testset)
    e3p_avg = e3p_sum / len(testset)
    return Metrics(train_time, rl_avg, epe_avg, e3p_avg)


@dataclass
class Metrics:
    time_taken: float = -1
    running_loss: float = -1
    epe: float = -1
    e3p: float = -1

    def __str__(self):
        res = f"Time taken: {self.time_taken:.2f}s\n"
        res += f"Running loss: {self.running_loss:.4f}\n"
        res += f"Endpoint error: {self.epe:.4f}\n"
        res += f"3 pixel error: {self.e3p:.4f}"
        return res
