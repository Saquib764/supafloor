"""Draw masks, polygon outlines, and points."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw


def draw_polygon_outline(
    image: Image.Image,
    polygon: list[list[int]],
    *,
    color: tuple[int, int, int] = (220, 50, 50),
    width: int = 2,
) -> Image.Image:
    overlay = image.convert("RGB").copy()
    if len(polygon) < 3:
        return overlay
    points = [(int(x), int(y)) for x, y in polygon]
    draw = ImageDraw.Draw(overlay)
    draw.line(points + [points[0]], fill=color, width=max(1, width))
    return overlay


def draw_mask(image: Image.Image, mask: np.ndarray, *, alpha: int = 120) -> Image.Image:
    composed = image.convert("RGBA")
    active = np.asarray(mask) > 0
    color = np.zeros((*active.shape, 4), dtype=np.uint8)
    color[active] = (0, 180, 60, alpha)
    return Image.alpha_composite(composed, Image.fromarray(color, mode="RGBA")).convert("RGB")


def draw_points(
    image: Image.Image,
    points: list[tuple[float, float] | list[int]],
    *,
    radius: int = 12,
    color: tuple[int, int, int] = (220, 50, 50),
) -> Image.Image:
    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    r = max(1, radius)
    for point in points:
        x, y = int(point[0]), int(point[1])
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=(44, 44, 44))
    return overlay
