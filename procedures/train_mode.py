from pickle import UnpicklingError

import torch
import torch.nn.functional as F
from data import DisparityDataset, index_set
from data.utils import pad_image_reverse, pad_image_to_multiple
from measures import error_3p, error_epe
from model import Net
from model.utils import choose_device, load_model, save_model
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import Logger, Metrics, create_logger


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
    iters_per_log: int,
    epochs_per_save: int,
    **kwargs,
):
    try:
        logger = create_logger(log_file, save_file, dataset_name, learning_rate)
    except ValueError as er:
        print(er)
        return

    # Choose device
    try:
        device = choose_device(cpu)
    except Exception as ex:
        print(ex)
        return

    # Create dataloaders
    try:
        trainset, testset = index_set(dataset_name, **kwargs)
        trainset = DisparityDataset(trainset)
        testset = DisparityDataset(testset, random_crop=False,crop_to_multiple=True)
        trainloader = DataLoader(
            trainset,
            batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        testloader = DataLoader(
            testset, 1, shuffle=False, num_workers=4, pin_memory=True
        )
    except ValueError as er:
        print(er)
        return

    # Create model, optimizer and grad scaler
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

    # Training and testing loop
    try:
        for epoch in range(epochs):
            print(f"Epoch {epoch+1} out of {epochs}")
            print("Training loop")

            trm = training_loop(
                net,
                trainloader,
                optimizer,
                scaler,
                device,
                epoch,
                iters_to_accumulate,
                logger,
                iters_per_log,
            )

            if save_file and epochs_per_save > 0 and (epoch + 1) % epochs_per_save == 0:
                save_model(net, optimizer, scaler, f"{save_file}-{epoch}.tmp")

            print("Training metrics:")
            print(trm)

            tm = Metrics()
            if eval_each_epoch > 0 and (epoch + 1) % eval_each_epoch == 0:
                print("Testing loop")
                tm = testing_loop(net, testloader, testset, device)
                print("Test metrics:")
                print(tm)

            logger.append("epoch", epoch, trm, tm)
    except KeyboardInterrupt:
        print("Received KeyboardInterrupt, closing")
    finally:
        if save_file:
            save_model(net, optimizer, scaler, save_file)
            print(f"Saved model to {save_file}")


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
    optimizer: Optimizer,
    scaler: GradScaler,
    device: torch.device,
    current_epoch: int,
    iters_to_accumulate: int,
    logger: Logger,
    iters_per_log: int,
):
    net.train()
    iter_metric = Metrics()
    epoch_metric = Metrics()
    for i, (left, right, gt) in tqdm(enumerate(trainloader), total=len(trainloader)):
        mask = torch.logical_and(gt < net.maxdisp, gt > 0)
        if len(gt[mask]) == 0:
            # Skip if mask covers whole image, since it would result in NaN loss
            # Low probability of happening, because of shuffled dataloader
            iter_metric.add(0, 0, 0, 0, 1)
            epoch_metric.add(0, 0, 0, 0, 1)
            continue
        left = left.to(device, non_blocking=True)
        right = right.to(device, non_blocking=True)
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
        items = gt.shape[0]
        loss = loss.item() * items
        epe = error_epe(gt[mask], d3[mask]) * items
        e3p = error_3p(gt[mask], d3[mask]) * items

        epoch_metric.add(loss, epe, e3p, items, 1)
        iter_metric.add(loss, epe, e3p, items, 1)

        if (iter_metric.iters % iters_per_log == 0) or (i + 1 == len(trainloader)):
            iter_metric.end()
            logger.append("iter", current_epoch, iter_metric)
            iter_metric = Metrics()

    epoch_metric.end()
    return epoch_metric


def testing_loop(
    net: Net, testloader: DataLoader, testset: DisparityDataset, device: torch.device
):
    net.eval()
    eval_metrics = Metrics()
    print("Evaluating on test set")
    for i, (left, right, gt) in tqdm(enumerate(testloader), total=len(testloader)):
        left = left.to(device, non_blocking=True)
        right = right.to(device, non_blocking=True)
        mask = torch.logical_and(gt < net.maxdisp, gt > 0)
        if len(gt[mask]) == 0:
            # Skip if mask covers whole image
            eval_metrics.add(0, 0, 0, 0, 1)
            continue

        gt = gt.to(device, non_blocking=True)
        # left, _ = pad_image_to_multiple(left)
        # right, p = pad_image_to_multiple(right)
        with torch.inference_mode():
            with autocast(device.type == "cuda"):
                d = net(left, right)
            # d = pad_image_reverse(d, p)

            items = gt.shape[0]
            epe = error_epe(gt[mask], d[mask]) * items
            e3p = error_3p(gt[mask], d[mask]) * items
            loss = F.smooth_l1_loss(d[mask], gt[mask]).item() * items
            # metrics
            eval_metrics.add(loss, epe, e3p, items, 1)
    eval_metrics.end()
    return eval_metrics
