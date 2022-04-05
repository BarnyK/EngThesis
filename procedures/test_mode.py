import enum
from data.dataset import DisparityDataset
from data import index_set
from torch.utils.data import DataLoader


def test_indexes(args: dict):
    try:
        train, test = index_set(**args)
        print(
            f"Dataset {args.get('dataset_name')!r} total images: {len(train)+len(test)} training: {len(train)}, test: {len(test)}"
        )
    except FileNotFoundError as er:
        print(er)
        print("The path specified is incorrect or the data inside is not as expected")
    except ValueError as er:
        print(er)
    except KeyError as er:
        print(er)
    except TypeError as er:
        print(er)


def print_validation(args: dict):
    try:
        train, test = index_set(**args)
        for t in test:
            print(t[2])
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
    from tqdm import tqdm
    import torch

    train, _ = index_set(**args)
    trainset = DisparityDataset(train, random_crop=False)
    trainloader = DataLoader(trainset, 1, shuffle=False, num_workers=2, pin_memory=True)
    for i, (l, r, d) in tqdm(enumerate(trainloader), total=len(trainloader)):
        if torch.min(d).item() <= 0:
            print(torch.min(d))
        if torch.isnan(l).any():
            print("nan detected", i)
        if torch.isnan(r).any():
            print("nan detected", i)
        if torch.isnan(d).any():
            print("nan detected", i)
