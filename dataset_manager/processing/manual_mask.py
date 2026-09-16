"""Load five-class structure masks from pre-rendered PNG files."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image

from .types import MaskChannels

_MASK_COLUMNS: tuple[tuple[str, str], ...] = (
    ("background", "background"),
    ("rooms", "room"),
    ("walls", "wall"),
    ("doors", "door"),
    ("windows", "window"),
)


def _resize_mask(image: Image.Image, size: int) -> np.ndarray:
    gray = np.asarray(image.convert("L"))
    if gray.shape != (size, size):
        gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_NEAREST)
    return (gray > 127).astype(np.float32)


def _resize_rgb(image: Image.Image, size: int) -> Image.Image:
    rgb = image.convert("RGB")
    if rgb.size == (size, size):
        return rgb
    return rgb.resize((size, size), Image.Resampling.LANCZOS)


def stack_mask_row(row: dict[str, Any], *, size: int, channels: MaskChannels) -> np.ndarray:
    layers: list[np.ndarray | None] = [None] * channels.count
    for channel_attr, row_key in _MASK_COLUMNS:
        index = getattr(channels, channel_attr)
        layers[index] = _resize_mask(row[row_key], size)
    if any(layer is None for layer in layers):
        missing = [
            row_key
            for (channel_attr, row_key) in _MASK_COLUMNS
            if layers[getattr(channels, channel_attr)] is None
        ]
        raise KeyError(f"row missing mask columns: {missing}")
    return np.stack(layers, axis=0)


def load_sample(
    row: dict[str, Any],
    *,
    size: int,
    channels: MaskChannels,
    image_key: str,
) -> tuple[Image.Image, np.ndarray]:
    """Hub row with PNG mask columns → RGB image and (C, H, W) mask targets."""
    image = _resize_rgb(row[image_key], size)
    masks = stack_mask_row(row, size=size, channels=channels)
    return image, masks
