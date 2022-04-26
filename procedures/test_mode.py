from data.dataset import DisparityDataset
from data import index_set
from torch.utils.data import DataLoader

"""
    Functions used for testing functionality of the network
"""

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


def print_training(args: dict):
    try:
        train, test = index_set(**args)
        for t in train:
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

    train, test = index_set(**args)
    trainset = DisparityDataset(train, random_crop=False, return_paths=True)
    testset = DisparityDataset(test, random_crop=False, return_paths=True)
    trainloader = DataLoader(
        trainset, 1, shuffle=False, num_workers=4, pin_memory=False
    )
    testloader = DataLoader(testset, 1, shuffle=False, num_workers=4, pin_memory=False)

    for loader in [trainloader, testloader]:
        for i, (l, r, d, p) in tqdm(enumerate(loader), total=len(loader)):
            pass
    print("Successfully loaded all files")
