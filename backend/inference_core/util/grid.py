"""Grid-based canvas point sampling."""

from __future__ import annotations

import random
from typing import Any

from PIL import Image


def sample_bright_grid_canvas_points(
    rng: random.Random,
    image: Image.Image,
    *,
    n: int,
    canvas_size: int,
    tries_per_cell: int = 5,
) -> list[tuple[float, float]]:
    gray = image.convert("L")
    pixels: Any = gray.load()
    width, height = gray.size
    cell_size = canvas_size / n
    points: list[tuple[float, float]] = []
    for row in range(n):
        for col in range(n):
            min_x = col * cell_size
            min_y = row * cell_size
            for _ in range(tries_per_cell):
                x = rng.uniform(min_x, min_x + cell_size)
                y = rng.uniform(min_y, min_y + cell_size)
                px = int(x)
                py = int(y)
                if 0 <= px < width and 0 <= py < height and pixels[px, py] >= 127:
                    points.append((x, y))
                    break
    return points
