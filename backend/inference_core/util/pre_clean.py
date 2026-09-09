"""Floor-plan image pre-cleaning for grid point sampling."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from util.render import draw_points


def _histogram_stretch_bounds(channel: np.ndarray) -> tuple[float, float]:
    hist = np.bincount(channel.astype(np.uint8).ravel(), minlength=256)
    below = hist[:127]
    above = hist[128:]
    pmin = float(np.argmax(below)) if below.any() else float(channel.min())
    pmax = float(np.argmax(above) + 128) if above.any() else float(channel.max())
    return pmin, pmax


def to_bw_stretched(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        gray = np.array(image.convert("L"), dtype=np.float64)
    else:
        arr = np.asarray(image)
        gray = np.array(Image.fromarray(arr).convert("L"), dtype=np.float64) if arr.ndim == 3 else arr.astype(np.float64)

    pmin, pmax = _histogram_stretch_bounds(gray)
    if pmax > pmin:
        gray = (gray - pmin) / (pmax - pmin) * 255.0
    else:
        gray = np.zeros_like(gray)
    return np.clip(gray, 0, 255).astype(np.uint8)


def pre_clean(image: Image.Image | np.ndarray) -> Image.Image:
    return Image.fromarray(to_bw_stretched(image), mode="L")


def sobel_edges(image: Image.Image) -> Image.Image:
    gray = np.array(image.convert("L"), dtype=np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    edges = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return Image.fromarray(edges, mode="L")


def find_corners(
    image: Image.Image,
    *,
    max_corners: int = 500,
    quality: float = 0.01,
    min_distance: int = 8,
) -> list[tuple[int, int]]:
    gray = np.array(image.convert("L"), dtype=np.uint8)
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners,
        qualityLevel=quality,
        minDistance=min_distance,
        blockSize=3,
        useHarrisDetector=False,
    )
    if corners is None:
        return []
    return [(int(x), int(y)) for x, y in corners[:, 0]]


def save_sobel(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sobel_edges(image).save(path)
    print(f"wrote {path}")


def save_corners_overlay(
    image: Image.Image,
    path: Path,
    *,
    corners: list[tuple[int, int]] | None = None,
) -> None:
    corner_pts = corners if corners is not None else find_corners(image)
    overlay = draw_points(image, corner_pts, radius=4, color=(220, 50, 50))
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(path)
    print(f"wrote {path}")
