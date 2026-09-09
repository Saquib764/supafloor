"""Draw a filled 5-point star prompt on floor-plan images."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw


def star_vertices(
    cx: float,
    cy: float,
    outer_r: float,
    *,
    inner_ratio: float = 0.4,
) -> list[tuple[float, float]]:
    inner_r = outer_r * inner_ratio
    points: list[tuple[float, float]] = []
    for index in range(10):
        angle = math.radians(-90 + index * 36)
        radius = outer_r if index % 2 == 0 else inner_r
        points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return points


def draw_filled_star(
    image: Image.Image,
    cx: float,
    cy: float,
    outer_r: float,
    *,
    color: tuple[int, int, int] = (255, 0, 0),
    inner_ratio: float = 0.4,
) -> None:
    ImageDraw.Draw(image).polygon(
        star_vertices(cx, cy, outer_r, inner_ratio=inner_ratio),
        fill=color,
    )


def star_inside_mask(
    mask: np.ndarray,
    cx: float,
    cy: float,
    outer_r: float,
    *,
    inner_ratio: float = 0.4,
) -> bool:
    height, width = mask.shape
    for x, y in star_vertices(cx, cy, outer_r, inner_ratio=inner_ratio):
        ix = int(round(x))
        iy = int(round(y))
        if ix < 0 or iy < 0 or ix >= width or iy >= height:
            return False
        if mask[iy, ix] <= 0:
            return False
    return True
