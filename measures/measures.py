from typing import Tuple
import torch
from time import time


@torch.jit.script
def error_3p(
    ground_truth: torch.Tensor,
    disparity: torch.Tensor,
    tau: Tuple[int, float] = (3, 0.05),
) -> float:
    """
    Calculate error for given disparity.
    Error is counted for ground truth pixels different than 0
    A pixel is counted as erronious if the difference
    to its true value exceeds 3 pixels and 5%.
    Taken from Kitty2015 evaluation metric
    """
    pixel_diff, percent_diff = tau
    mask = ground_truth > 0
    E = abs(disparity - ground_truth)[mask]
    n_err = len(E[(E > pixel_diff) & ((E / abs(ground_truth[mask])) > percent_diff)])
    return n_err / len(E)


@torch.jit.script
def error_epe(ground_truth: torch.Tensor, disparity: torch.Tensor) -> float:
    mask = ground_truth > 0
    res = torch.abs(disparity - ground_truth)[mask]
    return res.mean().item()


def test_error_3p():
    x = torch.arange(1, 21, 1).reshape((4, 5)).cuda()
    y = torch.clone(x)
    y[0, 0] = 5
    print(error_3p(y, x))


def test_error_epe():
    x = torch.arange(1, 21, 1, dtype=torch.float32).reshape((4, 5)).cuda()
    y = torch.clone(x)
    y[0, 0] = 5
    print(error_epe(y, x))


def test_time():
    gt = torch.rand((3, 256, 512), device="cuda")
    d = torch.rand((3, 256, 512), device="cuda")
    st = time()
    for i in range(100000):
        gt = torch.rand((3, 256, 512), device="cuda")
        d = torch.rand((3, 256, 512), device="cuda")
        error_epe(gt, d)
        error_3p(gt, d)
    et = time()
    print(et - st)


def test_time_():
    gt = torch.rand((3, 256, 512), device="cuda")
    d = torch.rand((3, 256, 512), device="cuda")
    st = time()
    for i in range(100000):
        gt = torch.rand((3, 256, 512), device="cuda")
        d = torch.rand((3, 256, 512), device="cuda")
        error_epe_(gt, d)
        error_3p_(gt, d)
    et = time()
    print(et - st)


if __name__ == "__main__":
    test_error_3p()
    test_error_epe()
    test_time()
    test_time_()
