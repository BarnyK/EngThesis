from typing import Tuple
import torch

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
    E = abs(disparity - ground_truth)
    n_err = len(E[(E > pixel_diff) & ((E / abs(ground_truth)) > percent_diff)])
    return n_err / len(E)


@torch.jit.script
def error_epe(
    ground_truth: torch.Tensor, disparity: torch.Tensor
) -> float:
    res = torch.abs(disparity - ground_truth)
    return res.mean().item()
