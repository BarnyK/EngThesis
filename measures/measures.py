from typing import Tuple
import torch
from time import time


@torch.jit.script
def error_3p(
    ground_truth: torch.Tensor,
    disparity: torch.Tensor,
    max_disp: int,
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
    mask = torch.logical_and(ground_truth <= max_disp, ground_truth > 0)
    E = abs(disparity - ground_truth)[mask]
    n_err = len(E[(E > pixel_diff) & ((E / abs(ground_truth[mask])) > percent_diff)])
    return n_err / len(E)


@torch.jit.script
def error_epe(ground_truth: torch.Tensor, disparity: torch.Tensor, max_disp: int) -> float:
    mask = torch.logical_and(ground_truth < max_disp, ground_truth > 0)
    res = torch.abs(disparity - ground_truth)[mask]
    return res.mean().item()
