"""Fit floor-plan images to a white RGB canvas."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def fit_to_canvas_rgb(image: Image.Image, *, size: int) -> Image.Image:
    image = image.convert("RGB")
    orig_w, orig_h = image.size
    scale = size / max(orig_w, orig_h)
    new_w = max(1, round(orig_w * scale))
    new_h = max(1, round(orig_h * scale))
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
    return canvas


def load_canvas_image(path: Path, *, canvas_size: int) -> Image.Image:
    return fit_to_canvas_rgb(Image.open(path), size=canvas_size)
