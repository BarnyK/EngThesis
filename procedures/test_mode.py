import enum
from data.dataset import DisparityDataset
from data.indexes import index_set
from torch.utils.data import DataLoader

def test_indexes(args: dict):
    try:
        train, test = index_set(**args)
        print(f"Dataset {args.get('name')!r} total images: {len(train)+len(test)} training: {len(train)}, test: {len(test)}")
        print("Test succeeded")
    except FileNotFoundError as er:
        print(er)
        print("The path specified is incorrect or the data inside is not as expected")
    except ValueError as er:
        print(er)
    except KeyError as er:
        print(er)
    except TypeError as er:
        print(er)

def test_loader(args: dict):
    train, _ = index_set(**args)
    trainset = DisparityDataset(train,random_crop=False)
    trainloader = DataLoader(trainset,1,shuffle=False,num_workers=0,pin_memory=True)
    for i,(l,r,d) in enumerate(trainloader):
        print(d.mean())