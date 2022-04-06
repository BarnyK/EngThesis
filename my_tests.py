import torch
from data.dataset import DisparityDataset
from data.indexing import index_set
from model import Net
from model.blocks import SDEABlock
import torch.nn.functional as F
from procedures.train_mode import prepare_model_optim_scaler

def main():
    net,opt,scaler = prepare_model_optim_scaler(None,torch.device("cuda"),192,False,0.001)
    left = torch.rand((3,3,256,512),device=torch.device("cuda"))
    right = torch.rand((3,3,256,512),device=torch.device("cuda"))
    gt = torch.rand((3,256,512),device=torch.device("cuda"))
    mask = gt > 0

    for i in range(10):
        opt.zero_grad()
        with torch.cuda.amp.autocast():
            net.train()
            d1,d2,d3 = net(left,right)
            loss = (
                    0.5 * F.smooth_l1_loss(d1[mask], gt[mask])
                    + 0.7 * F.smooth_l1_loss(d2[mask], gt[mask])
                    + F.smooth_l1_loss(d3[mask], gt[mask])
                )
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()


def main2():
    from procedures.evaluation_mode import eval_dataset
    kwargs = {
        "dataset_name":"monkaa",
        "root_images":"D:/datasets/monkaa__frames_cleanpass_webp",
        "root_disparity":"D:/datasets/monkaa__disparity",
        "split":0.2,
    }

    trainset, testset = index_set(**kwargs)
    trainset = DisparityDataset(trainset, random_crop=False,return_paths=True)
    testset = DisparityDataset(testset, random_crop=False,return_paths=True)
    return trainset, testset
t,tt = main2()