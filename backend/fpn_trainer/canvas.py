"""Render multi-channel room masks over floor-plan images."""

from __future__ import annotations

import numpy as np
from PIL import Image

def masks_to_rgb(masks: np.ndarray) -> Image.Image:
    """(C, H, W) float/bool → RGB PIL. R=boundary, G=doors, B=windows."""
    if masks.ndim != 3:
        raise ValueError(f"masks must be (C, H, W), got {masks.shape}")
    active = masks > 0.5
    height, width = active.shape[-2:]
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for channel in range(min(active.shape[0], 3)):
        rgb[..., channel] = (active[channel] * 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def overlay_rgb_mask(
    image: Image.Image,
    rgb_mask: Image.Image,
    *,
    alpha: float = 0.65,
) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    overlay = np.asarray(rgb_mask.convert("RGB"), dtype=np.float32)
    active = overlay.max(axis=-1, keepdims=True) > 0
    blended = base * (1.0 - alpha * active) + overlay * (alpha * active)
    return Image.fromarray(blended.clip(0, 255).astype(np.uint8), mode="RGB")


def overlay_masks(image: Image.Image, masks: np.ndarray, *, alpha: float = 0.65) -> Image.Image:
    return overlay_rgb_mask(image, masks_to_rgb(masks), alpha=alpha)


def merge_panels(panels: list[Image.Image]) -> Image.Image:
    if not panels:
        raise ValueError("panels must not be empty")
    width, height = panels[0].size
    merged = Image.new("RGB", (width * len(panels), height))
    for index, panel in enumerate(panels):
        merged.paste(panel.convert("RGB"), (index * width, 0))
    return merged
