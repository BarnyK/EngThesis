from os import path
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
    **kwargs,
):
    # torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(0)
    if save_file is not None:
        save_filename = path.basename(save_file)
        if log_file and path.isdir(log_file):
            log_file = path.join(log_file, f"{save_filename}.log")

    def save_log(epoch, t, rl, epe, e3p, epe_t, e3p_t):
        if log_file:
            with open(log_file, "a") as f:
                f.write(
                    f"{save_filename},{dataset_name},{learning_rate},{epoch},{t},{rl},{epe},{e3p},{epe_t},{e3p_t}\n"
                )

    # Index set
    # Create Dataset
    # Create Dataloader
    try:
        device = choose_device(cpu)
    except Exception as ex:
        print(ex)
        return

    trainset, testset = index_set(dataset_name, **kwargs)
    trainset = DisparityDataset(trainset)
    testset = DisparityDataset(testset, random_crop=False)

    trainloader = DataLoader(
        trainset,
        batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    testloader = DataLoader(testset, 1, shuffle=False, num_workers=2, pin_memory=False)

    net = Net(max_disp).to(device)

    if load_file:
        model_state, optim_state, scaler_state = load_model(load_file)
        net.load_state_dict(model_state)

    optimizer = torch.optim.Adam(net.parameters(), learning_rate, eps=1e-6)
    if load_file:
        optimizer.load_state_dict(optim_state)
        if learning_rate >= 0:
            optimizer.param_groups[0]["lr"] = learning_rate

    scaler = GradScaler(init_scale=256, enabled=not cpu)
    if load_file and scaler_state is not None:
        scaler.load_state_dict(scaler_state)

    for epoch in range(epochs):
        st = time()
        net.train()
        rl = 0.0
        epe_sum = 0
        e3p_sum = 0
        for i, (left, right, gt) in tqdm(
            enumerate(trainloader), total=len(trainloader)
        ):
            optimizer.zero_grad(False)
            left = left.to(device)
            right = right.to(device)
            mask = gt > 0
            gt = gt.to(device)

            with autocast(enabled=not cpu):
                d1, d2, d3 = net(left, right)
                loss = (
                    0.5 * F.smooth_l1_loss(d1[mask], gt[mask])
                    + 0.7 * F.smooth_l1_loss(d2[mask], gt[mask])
                    + F.smooth_l1_loss(d3[mask], gt[mask])
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Metrics
            epe_sum += error_epe(gt, d3) * d3.shape[0]
            e3p_sum += error_3p(gt, d3) * d3.shape[0]
            rl += loss.item() * d3.shape[0]

        train_time = time() - st
        rl_avg = rl / len(trainset)
        epe_avg = epe_sum / len(trainset)
        e3p_avg = e3p_sum / len(trainset)

        if save_file:
            save_model(net, optimizer, scaler, f"{save_file}-{epoch}.tmp")

        print(f"Epoch {epoch+1} out of {epochs}")
        print("Took: ", round(train_time, 2), "seconds")
        print("Running loss: ", rl_avg)
        print("EPE: ", epe_avg)
        print("3p error: ", e3p_avg)

        epe_sum_t = -1
        e3p_sum_t = -1
        if eval_each_epoch > 0 and (epoch + 1) % eval_each_epoch == 0:
            epe_sum_t = 0
            e3p_sum_t = 0
            print("Evaluating on test set")
            for i, (left, right, gt) in tqdm(
                enumerate(testloader), total=len(testloader)
            ):
                net.eval()
                left = left.to(device)
                right = right.to(device)
                gt = gt.to(device)
                left, og = pad_image(left)
                right, og = pad_image(right)
                with torch.inference_mode():
                    with torch.cuda.amp.autocast():
                        d = net(left, right)
                        d = pad_image_reverse(d, og)
                        epe = error_epe(gt, d) * d.shape[0]
                        e3p = error_3p(gt, d) * d.shape[0]
                epe_sum_t += epe
                e3p_sum_t += e3p
            epe_avg_t = epe_sum_t / len(testset)
            e3p_avg_t = e3p_sum_t / len(testset)
            print("Stats on test set:")
            print("EPE: ", epe_avg_t)
            print("E3P: ", e3p_avg_t)
        save_log(epoch, train_time, rl_avg, epe_avg, e3p_avg, epe_avg_t, e3p_avg_t)

    if save_file:
        save_model(net, optimizer, scaler, save_file)
