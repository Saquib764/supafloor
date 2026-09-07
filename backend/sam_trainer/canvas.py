"""Fit floor-plan images to canvas and render SAM masks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def fit_to_canvas(image: Image.Image, *, size: int, margin: int) -> Image.Image:
    image = image.convert("L")
    fit_size = size - 2 * margin
    orig_w, orig_h = image.size
    scale = fit_size / max(orig_w, orig_h)
    new_w = max(1, round(orig_w * scale))
    new_h = max(1, round(orig_h * scale))
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (size, size), 255)
    canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
    return canvas


def load_viewer_image(path: Path, *, canvas_size: int, margin: int) -> Image.Image:
    return fit_to_canvas(Image.open(path), size=canvas_size, margin=margin).convert("RGB")


def apply_masks(
    image: Image.Image,
    masks: list[np.ndarray],
    points: list[tuple[float, float]],
) -> Image.Image:
    rgb = image.convert("RGB")
    if not masks:
        return rgb
    composed = rgb.convert("RGBA")
    for mask in masks:
        color = np.zeros((*mask.shape, 4), dtype=np.uint8)
        color[mask] = (0, 180, 60, 120)
        composed = Image.alpha_composite(composed, Image.fromarray(color, mode="RGBA"))
    draw = ImageDraw.Draw(composed)
    radius = 6
    for x, y in points:
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(220, 50, 50, 255),
            outline=(44, 44, 44, 255),
            width=2,
        )
    return composed.convert("RGB")
