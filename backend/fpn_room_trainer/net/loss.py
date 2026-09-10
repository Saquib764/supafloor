"""Mask segmentation loss (dice + focal)."""

from __future__ import annotations

from torch import Tensor
from torch.nn import functional as F


def dice_loss(logits: Tensor, targets: Tensor, *, eps: float = 1e-6) -> Tensor:
    probs = F.sigmoid(logits)
    num = 2.0 * (probs * targets).sum(dim=(-2, -1))
    den = probs.sum(dim=(-2, -1)) + targets.sum(dim=(-2, -1))
    return (1.0 - (num + eps) / (den + eps)).mean()


def focal_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = F.sigmoid(logits)
    prob_t = probs * targets + (1.0 - probs) * (1.0 - targets)
    return (alpha * (1.0 - prob_t).pow(gamma) * bce).mean()


def mask_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    focal_alpha: float,
    focal_gamma: float,
) -> Tensor:
    if logits.shape[-2:] != targets.shape[-2:]:
        targets = F.interpolate(
            targets.unsqueeze(1),
            size=logits.shape[-2:],
            mode="nearest",
        ).squeeze(1)
    return dice_loss(logits, targets) + focal_loss(
        logits,
        targets,
        alpha=focal_alpha,
        gamma=focal_gamma,
    )


def room_mask_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    focal_alpha: float,
    focal_gamma: float,
) -> Tensor:
    if logits.ndim != 4 or targets.ndim != 4:
        raise ValueError(f"logits and targets must be (B, 1, H, W), got {logits.shape}, {targets.shape}")
    if logits.shape[1] != 1 or targets.shape[1] != 1:
        raise ValueError(
            f"expected single-channel masks, got logits {logits.shape[1]}, targets {targets.shape[1]}"
        )
    return mask_loss(
        logits[:, 0],
        targets[:, 0],
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
    )
