import numpy as np
from typing import Tuple
from typing import Union
import torch


def error_3p(
    ground_truth: np.ndarray, disparity: np.ndarray, tau: Tuple[int, float] = (3, 0.05)
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


def error_epe(ground_truth: np.ndarray, disparity: np.ndarray) -> float:
    mask = ground_truth > 0
    res = sum(abs(disparity - ground_truth)[mask])
    if isinstance(res, torch.Tensor):
        return res.item()
    return res


def test_error_3p():
    x: np.ndarray = np.arange(20)
    x = x.reshape((4, 5)) + 1
    y = x.copy()
    y[0, 0] = 5
    print(error_3p(y, x))


def test_error_3p_t():
    x = torch.arange(1, 21, 1).reshape((4, 5)).cuda()
    y = torch.clone(x)
    y[0, 0] = 5
    print(error_3p(y, x))


def test_error_epe():
    x: np.ndarray = np.arange(20, dtype=np.float32)
    x = x.reshape((4, 5)) + 1
    y = x.copy()
    y[0, 0] = 5
    print(error_epe(y, x))


def test_error_epe_t():
    x = torch.arange(1, 21, 1, dtype=torch.float32).reshape((4, 5)).cuda()
    y = torch.clone(x)
    y[0, 0] = 5
    print(error_epe(y, x))


if __name__ == "__main__":
    test_error_3p()
    test_error_3p_t()
    test_error_epe()
    test_error_epe_t()
