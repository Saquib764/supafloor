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


def _extra_opening_penalty(
    logits: Tensor,
    targets: Tensor,
    opening_masks: Tensor,
    *,
    extra_opening_penalty: float,
) -> Tensor:
    opening = _align_spatial(opening_masks, logits.shape[-2:])
    opening_mass = opening.sum()
    if opening_mass <= 0:
        return logits.new_tensor(0.0)
    probs = F.sigmoid(logits)
    false_negative = ((1.0 - probs) * opening * targets).sum() / opening_mass
    return extra_opening_penalty * false_negative


def structure_mask_loss(
    logits: Tensor,
    targets: Tensor,
    opening_masks: Tensor,
    *,
    focal_alpha: float,
    focal_gamma: float,
    channel_weights: tuple[float, ...],
    extra_opening_penalty: float,
) -> Tensor:
    if logits.ndim != 4 or targets.ndim != 4:
        raise ValueError(f"logits and targets must be (B, C, H, W), got {logits.shape}, {targets.shape}")
    channels = logits.shape[1]
    if targets.shape[1] != channels:
        raise ValueError(f"channel mismatch: logits {channels}, targets {targets.shape[1]}")
    if len(channel_weights) != channels:
        raise ValueError(f"channel_weights length ({len(channel_weights)}) != channels ({channels})")
    if opening_masks.ndim != 3:
        raise ValueError(f"opening_masks must be (B, H, W), got {opening_masks.shape}")
    if extra_opening_penalty < 0.0:
        raise ValueError(f"extra_opening_penalty ({extra_opening_penalty}) must be ≥ 0")

    total = logits.new_tensor(0.0)
    for index in range(channels):
        channel_loss = mask_loss(
            logits[:, index],
            targets[:, index],
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
        )
        if index == 0 and extra_opening_penalty > 0.0:
            channel_loss = channel_loss + _extra_opening_penalty(
                logits[:, index],
                targets[:, index],
                opening_masks,
                extra_opening_penalty=extra_opening_penalty,
            )
        total = total + channel_weights[index] * channel_loss
    return total
