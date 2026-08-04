"""Resize floor plan images onto a fixed square canvas."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from extraction.constants import CANVAS_SIZE, MORPH_ITERATIONS, MORPH_KERNEL_SIZE, THRESHOLD_VALUE


def _load_image(source: bytes | Path | str) -> Image.Image:
    if isinstance(source, (str, Path)):
        return Image.open(source).convert("RGB")
    return Image.open(io.BytesIO(source)).convert("RGB")


def _to_bw_morph(rgb: Image.Image) -> Image.Image:
    """Grayscale → erosion → dilation (remove thin lines) → binary threshold."""
    gray = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))

    step = gray
    # step = cv2.dilate(step, kernel, iterations=MORPH_ITERATIONS)
    # step = cv2.erode(step, kernel, iterations=MORPH_ITERATIONS)

    # step = cv2.erode(step, kernel, iterations=MORPH_ITERATIONS)
    # step = cv2.dilate(step, kernel, iterations=MORPH_ITERATIONS)
    _, bw = cv2.threshold(step, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
    return Image.fromarray(bw, mode="L").convert("RGB")


def prepare_canvas(source: bytes | Path | str) -> tuple[bytes, Image.Image, dict[str, Any]]:
    """Fit image inside CANVAS_SIZE, center on white square canvas. Returns PNG bytes, PIL image, metadata."""
    img = _load_image(source)
    orig_w, orig_h = img.size
    scale = min(1.0, CANVAS_SIZE / max(orig_w, orig_h))
    new_w = max(1, round(orig_w * scale))
    new_h = max(1, round(orig_h * scale))
    resized = img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), "white")
    offset_x = (CANVAS_SIZE - new_w) // 2
    offset_y = (CANVAS_SIZE - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))
    canvas = _to_bw_morph(canvas)

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    meta = {
        "canvas_size": CANVAS_SIZE,
        "original_size": [orig_w, orig_h],
        "scaled_size": [new_w, new_h],
        "scale": scale,
        "offset_x": offset_x,
        "offset_y": offset_y,
        "bw": True,
        "morph_kernel_size": MORPH_KERNEL_SIZE,
        "morph_iterations": MORPH_ITERATIONS,
        "threshold": THRESHOLD_VALUE,
    }
    return buf.getvalue(), canvas, meta
