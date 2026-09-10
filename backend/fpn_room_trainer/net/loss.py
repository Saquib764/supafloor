"""Mask segmentation loss (dice + focal)."""

from __future__ import annotations

from torch import Tensor
from torch.nn import functional as F


def _align_spatial(mask: Tensor, size: tuple[int, int]) -> Tensor:
    if mask.shape[-2:] == size:
        return mask
    return F.interpolate(mask.unsqueeze(1), size=size, mode="nearest").squeeze(1)


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
        targets = _align_spatial(targets, logits.shape[-2:])
    return dice_loss(logits, targets) + focal_loss(
        logits,
        targets,
        alpha=focal_alpha,
        gamma=focal_gamma,
    )


def room_mask_loss(
    logits: Tensor,
    targets: Tensor,
    opening_masks: Tensor,
    *,
    focal_alpha: float,
    focal_gamma: float,
    extra_opening_penalty: float,
) -> Tensor:
    if logits.ndim != 4 or targets.ndim != 4:
        raise ValueError(f"logits and targets must be (B, 1, H, W), got {logits.shape}, {targets.shape}")
    if logits.shape[1] != 1 or targets.shape[1] != 1:
        raise ValueError(
            f"expected single-channel masks, got logits {logits.shape[1]}, targets {targets.shape[1]}"
        )
    if opening_masks.ndim != 3:
        raise ValueError(f"opening_masks must be (B, H, W), got {opening_masks.shape}")
    if extra_opening_penalty < 0.0:
        raise ValueError(f"extra_opening_penalty ({extra_opening_penalty}) must be ≥ 0")

    logits_2d = logits[:, 0]
    targets_2d = targets[:, 0]
    base = mask_loss(
        logits_2d,
        targets_2d,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
    )
    if extra_opening_penalty <= 0.0:
        return base

    opening = _align_spatial(opening_masks, logits_2d.shape[-2:])
    opening_mass = opening.sum()
    if opening_mass <= 0:
        return base

    probs = F.sigmoid(logits_2d)
    false_positive = (probs * opening * (1.0 - targets_2d)).sum() / opening_mass
    return base + extra_opening_penalty * false_positive
