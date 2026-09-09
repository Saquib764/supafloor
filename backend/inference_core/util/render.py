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


_MASK_COLORS = (
    (0, 180, 60),
    (220, 50, 50),
    (50, 120, 220),
    (230, 160, 40),
    (160, 60, 220),
    (40, 180, 180),
)


def draw_mask(
    image: Image.Image,
    mask: np.ndarray,
    *,
    alpha: int = 120,
    color: tuple[int, int, int] = (0, 180, 60),
) -> Image.Image:
    composed = image.convert("RGBA")
    active = np.asarray(mask) > 0
    layer = np.zeros((*active.shape, 4), dtype=np.uint8)
    layer[active] = (*color, alpha)
    return Image.alpha_composite(composed, Image.fromarray(layer, mode="RGBA")).convert("RGB")


def draw_masks(
    image: Image.Image,
    masks: list[np.ndarray],
    *,
    alpha: int = 120,
) -> Image.Image:
    composed = image.convert("RGBA")
    for index, mask in enumerate(masks):
        active = np.asarray(mask) > 0
        color = _MASK_COLORS[index % len(_MASK_COLORS)]
        layer = np.zeros((*active.shape, 4), dtype=np.uint8)
        layer[active] = (*color, alpha)
        composed = Image.alpha_composite(composed, Image.fromarray(layer, mode="RGBA"))
    return composed.convert("RGB")


def render_image_diff(before: Image.Image, after: Image.Image) -> Image.Image:
    """Absolute per-pixel RGB difference, scaled for visibility."""
    before_rgb = before.convert("RGB")
    after_rgb = after.convert("RGB")
    if after_rgb.size != before_rgb.size:
        after_rgb = after_rgb.resize(before_rgb.size, Image.Resampling.LANCZOS)

    diff = np.abs(
        np.asarray(after_rgb, dtype=np.int16) - np.asarray(before_rgb, dtype=np.int16)
    ).astype(np.uint8)
    peak = int(diff.max())
    if 0 < peak < 64:
        diff = np.clip(diff * (255 // peak), 0, 255).astype(np.uint8)
    return Image.fromarray(diff, mode="RGB")


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
