"""Render multi-channel floor-plan masks for preview and viz."""

from __future__ import annotations

import numpy as np
from PIL import Image


def masks_to_rgb(masks: np.ndarray) -> Image.Image:
    """(C, H, W) float/bool → RGB PIL. R=walls, G=doors, B=windows."""
    if masks.ndim != 3:
        raise ValueError(f"masks must be (C, H, W), got {masks.shape}")
    active = masks > 0.5
    height, width = active.shape[-2:]
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for channel in range(min(active.shape[0], 3)):
        rgb[..., channel] = (active[channel] * 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def mask_channel_to_rgb(masks: np.ndarray, channel: int) -> Image.Image:
    """Render one mask channel in its viz color (R/G/B)."""
    if masks.ndim != 3:
        raise ValueError(f"masks must be (C, H, W), got {masks.shape}")
    if channel < 0 or channel >= masks.shape[0]:
        raise ValueError(f"channel {channel} out of range for {masks.shape[0]} channels")
    active = masks[channel] > 0.5
    height, width = active.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    if channel < 3:
        rgb[..., channel] = (active * 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def merge_panels(panels: list[Image.Image]) -> Image.Image:
    if not panels:
        raise ValueError("panels must not be empty")
    width, height = panels[0].size
    merged = Image.new("RGB", (width * len(panels), height))
    for index, panel in enumerate(panels):
        merged.paste(panel.convert("RGB"), (index * width, 0))
    return merged


def side_by_side_mask(image: Image.Image, masks: np.ndarray) -> Image.Image:
    """Floor plan and full multi-channel mask RGB, side by side."""
    return merge_panels([image.convert("RGB"), masks_to_rgb(masks)])


def side_by_side_mask_channel(image: Image.Image, masks: np.ndarray, channel: int) -> Image.Image:
    """Floor plan and one mask channel, side by side."""
    return merge_panels([image.convert("RGB"), mask_channel_to_rgb(masks, channel)])
