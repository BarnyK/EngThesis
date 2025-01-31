from data import index_set
from data.dataset import DisparityDataset
from torch.utils.data import DataLoader

"""
    Functions used for testing functionality of the network
"""


def test_indexes(args: dict):
    try:
        train, test, disp_func = index_set(**args)
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
        train, test, disp_func = index_set(**args)
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
        train, test, disp_func = index_set(**args)
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
    import torch
    from tqdm import tqdm

    train, test, disp_func = index_set(**args)
    trainset = DisparityDataset(train, disp_func, random_crop=False, return_paths=True)
    testset = DisparityDataset(test, disp_func, random_crop=False, return_paths=True)
    trainloader = DataLoader(
        trainset, 1, shuffle=False, num_workers=4, pin_memory=False
    )
    testloader = DataLoader(testset, 1, shuffle=False, num_workers=4, pin_memory=False)

    for loader in [trainloader, testloader]:
        for i, (l, r, d, p) in tqdm(enumerate(loader), total=len(loader)):
            pass
    print("Successfully loaded all files")
