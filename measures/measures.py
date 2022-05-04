from typing import Tuple

import torch
import torch.nn.functional as F


@torch.jit.script
def error_3p(
    ground_truth: torch.Tensor,
    disparity: torch.Tensor,
    tau: Tuple[int, float] = (3, 0.05),
) -> float:
    """
    3 pixel error, defined as percent of bad pixels in an image
    """
    pixel_diff, percent_diff = tau
    E = abs(disparity - ground_truth)
    n_err = len(E[(E > pixel_diff) & ((E / abs(ground_truth)) > percent_diff)])
    return n_err / len(E)


@torch.jit.script
def error_epe(ground_truth: torch.Tensor, disparity: torch.Tensor) -> float:
    res = torch.abs(disparity - ground_truth)
    return res.mean().item()


def loss(ground_truth: torch.Tensor, pred3, pred2=None, pred1=None):
    if pred2 is not None and pred1 is not None:
        return (
            0.5 * F.smooth_l1_loss(pred1, ground_truth)
            + 0.7 * F.smooth_l1_loss(pred2, ground_truth)
            + F.smooth_l1_loss(pred3, ground_truth)
        )
    return F.smooth_l1_loss(pred3, ground_truth)
