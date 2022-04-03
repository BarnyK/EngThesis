from statistics import mode

import torch
from tqdm import tqdm
from data.dataset import DisparityDataset, pad_image, pad_image_reverse
from data.indexes import index_set
from torch.utils.data import DataLoader
from time import time
from measures.measures import error_3p, error_epe
from model.utils import choose_device,load_model,save_model
from model import Net
import torch.nn.functional as F
def train(epochs,batch_size,learning_rate,dataset_name,cpu,max_disp,load_file,save_file,**kwargs):
    # Index set
    # Create Dataset
    # Create Dataloader
    try:
        device = choose_device(cpu)
    except Exception as ex:
        print(ex)
        return

    trainset, testset = index_set(dataset_name,**kwargs)
    trainset = DisparityDataset(trainset)
    testset = DisparityDataset(testset,random_crop=False)

    trainloader = DataLoader(trainset,batch_size,shuffle=True,num_workers=2,pin_memory=True)
    testloader = DataLoader(testset,1,shuffle=False,num_workers=2,pin_memory=False)
        
    net = Net(max_disp).to(device)
    
    if load_file:
        model_state, optim_state = load_model(load_file)
        net.load_state_dict(model_state)
    
    optimizer = torch.optim.Adam(net.parameters(),learning_rate)
    if load_file:
        optimizer.load_state_dict(optim_state)
        if learning_rate >= 0:
            optimizer.param_groups[0]['lr'] = learning_rate

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        st = time()
        net.train()
        rl = 0.0
        for i,(left,right,gt) in tqdm(enumerate(trainloader),total=len(trainloader)):
            optimizer.zero_grad()
            left = left.to(device)
            right = right.to(device)
            gt = gt.to(device)
            with torch.cuda.amp.autocast():
                d1,d2,d3 = net(left,right) 
                mask = gt > 0
                loss = (
                    0.5 * F.smooth_l1_loss(d1[mask], gt[mask], size_average=True)
                    + 0.7 * F.smooth_l1_loss(d2[mask], gt[mask], size_average=True)
                    + F.smooth_l1_loss(d3[mask], gt[mask], size_average=True)
                )
            scaler.scale(loss).backward()
            loss.backward()
            optimizer.step()
            rl += loss.item()
        et = time()

        if save_file:
            save_model(net,optimizer,save_file+f"-{epoch}")

        
        epe_sum = 0
        e3p_sum = 0
        for i,(left,right,gt) in tqdm(enumerate(testloader)):
            net.eval()
            left = left.to(device)
            right = right.to(device)
            gt = gt.to(device)
            left,og = pad_image(left)
            right,og = pad_image(right)
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    d = net(left,right)
                    d = pad_image_reverse(d,og)
                    epe = error_epe(gt,d)
                    e3p = error_3p(gt,d)
            epe_sum += epe
            e3p_sum += e3p
        epe_avg = epe_sum/(i+1)
        e3p_avg = e3p_sum/(i+1)
        print(f"Epoch {epoch} out of {epochs}")
        print("Took: ", round(et-st,2))
        print("RL: ",rl/i)
        print("EPE: ",epe_avg)
        print("E3P: ",e3p_avg)
    if save_file:
        save_model(net,optimizer,save_file)