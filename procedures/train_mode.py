from statistics import mode

import torch
from data.dataset import DisparityDataset
from data.indexes import index_set
from torch.utils.data import DataLoader
from time import time
from model.utils import choose_device,load_model,save_model
from model import Net

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

    trainloader = DataLoader(trainset,batch_size,shuffle=True,num_workers=1,pin_memory=True)
    testloader = DataLoader(testset,batch_size,shuffle=False,num_workers=1,pin_memory=False)
        
    net = Net(max_disp).to(device)
    net2 = Net(256)

    net.load_state_dict(net2.state_dict())

    optimizer = torch.optim.Adam(net.parameters(),learning_rate)

    if load_file:
        model_state, optim_state, loaded_max_disp = load_model(load_file)




    net.train()

    for epoch in range(epochs):
        st = time()
        for i,(left,right,gt) in enumerate(trainloader):
            left = left.to(device)
            right = right.to(device)
            gt = gt.to(device)
            
            d1,d2,d3 = net(left,right)  


    st = time()
      
    et = time()
    print(et-st,"pin_memory=True")
