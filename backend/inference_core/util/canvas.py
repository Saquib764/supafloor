"""Fit floor-plan images to a white RGB canvas."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def fit_to_canvas_rgb(image: Image.Image, *, size: int) -> Image.Image:
    image = image.convert("RGB")
    orig_w, orig_h = image.size
    paste_x, paste_y, new_w, new_h = fit_to_canvas_placement(orig_w, orig_h, size=size)
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(resized, (paste_x, paste_y))
    return canvas


def fit_to_canvas_placement(orig_w: int, orig_h: int, *, size: int) -> tuple[int, int, int, int]:
    """Return ``paste_x, paste_y, content_w, content_h`` for fit_to_canvas_rgb."""
    scale = size / max(orig_w, orig_h)
    content_w = max(1, round(orig_w * scale))
    content_h = max(1, round(orig_h * scale))
    paste_x = (size - content_w) // 2
    paste_y = (size - content_h) // 2
    return paste_x, paste_y, content_w, content_h


def load_canvas_image(path: Path, *, canvas_size: int) -> Image.Image:
    return fit_to_canvas_rgb(Image.open(path), size=canvas_size)
