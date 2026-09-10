"""Render multi-channel floor-plan masks for preview and viz."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

def masks_to_rgb(masks: np.ndarray) -> Image.Image:
    """(C, H, W) float/bool → RGB PIL. First three channels on R/G/B."""
    if masks.ndim != 3:
        raise ValueError(f"masks must be (C, H, W), got {masks.shape}")
    active = masks > 0.5
    height, width = active.shape[-2:]
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for channel in range(min(active.shape[0], 3)):
        rgb[..., channel] = (active[channel] * 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def to_gray_image(image: Image.Image) -> Image.Image:
    """Single-channel view for preview panels."""
    return image.convert("L")


def mask_channel_to_gray(
    masks: np.ndarray,
    channel: int,
    *,
    background_channel: int | None = None,
) -> Image.Image:
    """One mask channel as grayscale (white = active)."""
    if masks.ndim != 3:
        raise ValueError(f"masks must be (C, H, W), got {masks.shape}")
    if channel < 0 or channel >= masks.shape[0]:
        raise ValueError(f"channel {channel} out of range for {masks.shape[0]} channels")
    if channel == background_channel:
        gray = np.full(masks.shape[-2:], 255, dtype=np.uint8)
        gray[masks[channel] <= 0.5] = 0
        return Image.fromarray(gray, mode="L")
    active = (masks[channel] > 0.5).astype(np.uint8) * 255
    return Image.fromarray(active, mode="L")


def merge_panels(panels: list[Image.Image]) -> Image.Image:
    if not panels:
        raise ValueError("panels must not be empty")
    mode = panels[0].mode
    width, height = panels[0].size
    fill = 255 if mode == "L" else (0, 0, 0)
    merged = Image.new(mode, (width * len(panels), height), fill)
    for index, panel in enumerate(panels):
        merged.paste(panel.convert(mode), (index * width, 0))
    return merged


def stack_rows(rows: list[Image.Image]) -> Image.Image:
    if not rows:
        raise ValueError("rows must not be empty")
    mode = rows[0].mode
    width = max(row.width for row in rows)
    height = sum(row.height for row in rows)
    fill = 255 if mode == "L" else (255, 255, 255)
    merged = Image.new(mode, (width, height), fill)
    y = 0
    for row in rows:
        merged.paste(row.convert(mode), (0, y))
        y += row.height
    return merged


def with_label(image: Image.Image, label: str) -> Image.Image:
    labeled = image.copy()
    draw = ImageDraw.Draw(labeled)
    draw.rectangle((0, 0, min(labeled.width, 360), 28), fill=255)
    draw.text((8, 6), label, fill=0)
    return labeled


def columns_input_masks(
    image: Image.Image,
    masks: np.ndarray,
    *,
    mask_labels: tuple[str, ...] | None = None,
    background_channel: int | None = None,
) -> Image.Image:
    """Input plus one grayscale column per mask channel."""
    panels = [with_label(to_gray_image(image), "input")]
    for channel in range(masks.shape[0]):
        name = mask_labels[channel] if mask_labels and channel < len(mask_labels) else f"ch{channel}"
        panels.append(
            with_label(
                mask_channel_to_gray(masks, channel, background_channel=background_channel),
                name,
            )
        )
    return merge_panels(panels)


def side_by_side_mask(image: Image.Image, masks: np.ndarray) -> Image.Image:
    """Floor plan and full multi-channel mask RGB, side by side."""
    return merge_panels([image.convert("RGB"), masks_to_rgb(masks)])
