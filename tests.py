import torch
from torch import nn
from torch import functional as F
from data.indexes import index_kitti2012
from model import Net


# def single_pass(model, trainloader, evalloader, optimizer, loss_function):
#     print(model)
#     for i, (left, right, disp) in enumerate(trainloader):
#         left, right, disp = left.cuda(), right.cuda(), disp.cuda()
#         optimizer.zero_grad()

#         with torch.cuda.amp_autocast():
#             p1, p2, p3 = model(left, right)


# def train(
#     model: nn.Module,
#     optimizer: torch.optim.optimizer.Optimizer,
#     left: torch.Tensor,
#     right: torch.Tensor,
#     disp: torch.Tensor,
# ):
#     model.train()

#     mask = disp > 0
#     mask.detach_()
#     optimizer.zero_grad()
#     with torch.cuda.amp.autocast():
#         p1, p2, p3 = model(left, right)

#     p1 = torch.squeeze(p1, 1)
#     p2 = torch.squeeze(p2, 1)
#     p3 = torch.squeeze(p3, 1)
    
#     loss = (
#         0.5 * F.smooth_l1_loss(p1[mask], disp[mask], size_average=True)
#         + 0.7 * F.smooth_l1_loss(p2[mask], disp[mask], size_average=True)
#         + F.smooth_l1_loss(p3[mask], disp[mask], size_average=True)
#     )

#     loss.backward()
#     # optimizer.step()
#     return loss.data



def main(max_disp=192,gpu=True):
    if gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            raise Exception("gpu usage was specified, but a device was not found")
    else:
        device = torch.device("cpu")
    train_data, test_data = index_kitti2012("E:\\Thesis\\Datasets\\Kitty2012\\data_stereo_flow\\training")
    net = Net(max_disp).to(device)

if __name__ == "__main__":
    main()