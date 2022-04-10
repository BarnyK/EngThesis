import torch
from data.dataset import DisparityDataset, read_and_prepare
from data.indexing import index_set
from data.utils import pad_image, pad_image_reverse
from measures.measures import error_3p
from model import Net
from model.blocks import SDEABlock
import torch.nn.functional as F
from procedures.train_mode import prepare_model_optim_scaler


def main():
    torch.manual_seed(1111)
    device = torch.device("cpu")
    net, _, _ = prepare_model_optim_scaler(None, device, 192, False, 0.001)
    left = torch.rand((3, 3, 256, 512), device=device)
    right = torch.rand((3, 3, 256, 512), device=device)
    gt = torch.rand((3, 256, 512), device=device)
    mask = torch.logical_and(gt < net.maxdisp, gt > 0)

    # for i in range(1):
    d1, d2, d3 = net(left, right)
    loss = (
        0.5 * F.smooth_l1_loss(d1[mask], gt[mask])
        + 0.7 * F.smooth_l1_loss(d2[mask], gt[mask])
        + F.smooth_l1_loss(d3[mask], gt[mask])
    )
    print(loss)
    net.eval()
    d = net(left, right)
    loss = F.smooth_l1_loss(d[mask], gt[mask])
    e3p = error_3p(gt, d, 192)
    print(loss.item(), e3p)


def main2():
    from procedures.evaluation_mode import eval_dataset

    kwargs = {
        "dataset_name": "monkaa",
        "root_images": "/home/barny/Desktop/datasets/sceneflow/monkaa__frames_cleanpass_webp",
        "root_disparity": "/home/barny/Desktop/datasets/sceneflow/monkaa__disparity",
        "split": 0.2,
    }

    trainset, testset = index_set(**kwargs)
    trainset = DisparityDataset(trainset, random_crop=False, return_paths=True)
    testset = DisparityDataset(testset, random_crop=False, return_paths=True)
    from torch.utils.data import DataLoader

    trainloader = DataLoader(trainset, 1)
    for i, (l, r, d, p) in enumerate(trainloader):
        print(l.shape)
        print(r.shape)
        print(d.shape)
        break
    ll, rr, dd = read_and_prepare(p[0][0], p[1][0], p[2][0])
    print(ll.shape)
    print(rr.shape)
    print(dd.shape)
    tloader = DataLoader([(ll, rr, dd), (ll, rr, dd)], 1)
    for i, (z) in enumerate(trainloader):
        print(z[0].shape)
        print(z[1].shape)
        print(z[2].shape)
        break


def main3():
    left = torch.ones((1, 1, 2, 2))
    lp, s = pad_image(left)
    lpp = pad_image_reverse(lp, s)
    print(left)
    print(lp)
    print(lpp)


def main4():
    from procedures.evaluation_mode import evaluate_one

    evaluate_one(
        "/home/barny/Desktop/datasets/sceneflow/flyingthings3d__frames_cleanpass_webp/frames_cleanpass_webp/TEST/A/0149/left/0012.webp",
        "/home/barny/Desktop/datasets/sceneflow/flyingthings3d__frames_cleanpass_webp/frames_cleanpass_webp/TEST/A/0149/right/0012.webp",
        "/home/barny/Desktop/datasets/kittis/samples/garbo.png",
        "/home/barny/Desktop/datasets/sceneflow/flyingthings3d__disparity/disparity/TEST/A/0149/left/0012.pfm",
        192,
        "/home/barny/Desktop/models/sceneflow-1-1-3.tmp",
        True,
        False,
    )


def main5():
    from data.utils import pad_image_, pad_image_reverse_

    left = torch.ones((1, 1, 3, 3))
    lp, s = pad_image_(left)
    lpp = pad_image_reverse_(lp, s)
    print(left.shape)
    print(lp.shape)
    print(lp)
    print(torch.abs(left - lpp).sum())


def main5():
    from procedures.evaluation_mode import eval_dataset

    eval_dataset(
        "kittis",
        192,
        True,
        True,
        None,
        "/home/barny/Desktop/test.log",
        True,
        root="/home/barny/Desktop/datasets/kittis",
        split=1,
    )


def main6():
    from model import Net
    from torchinfo import summary

    with torch.cuda.amp.autocast():
        x = Net(192, False).cuda()
        data = summary(x, [(3, 3, 256, 512), (3, 3, 256, 512)], depth=10, device="cuda")
        with open("summary2.txt", "w", encoding="utf-8") as f:
            f.write(str(data))


main6()
