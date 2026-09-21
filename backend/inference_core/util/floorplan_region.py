"""Derive a floorplan region mask and boundary polygon from fpn_five background."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image



@dataclass(frozen=True)
class FloorplanRegion:
    mask: np.ndarray
    polygon: list[list[int]]


def _drop_small_components(binary: np.ndarray, *, min_area_px: int) -> np.ndarray:
    if min_area_px <= 0 or not binary.any():
        return binary
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    kept = np.zeros_like(binary)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area_px:
            kept[labels == label] = 1
    return kept


def compute_floorplan_region(
    background: np.ndarray,
    *,
    close_px: int,
    approx_epsilon: float,
    dilate_px: int = 0,
    min_component_area_ratio: float = 0.02,
) -> FloorplanRegion:
    """ROI = invert background; close holes, keep largest patch, fit polygon."""
    plane = background[0] if background.ndim == 3 else background
    binary = (~(plane > 0.5)).astype(np.uint8)
    if not binary.any():
        return FloorplanRegion(
            mask=np.zeros_like(binary, dtype=bool),
            polygon=[],
        )

    if dilate_px > 0:
        k = 2 * dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        binary = cv2.dilate(binary, kernel)

    if close_px > 0:
        k = 2 * close_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    min_area_px = int(min_component_area_ratio * binary.shape[0] * binary.shape[1])
    binary = _drop_small_components(binary, min_area_px=min_area_px)
    if not binary.any():
        return FloorplanRegion(mask=np.zeros_like(binary, dtype=bool), polygon=[])

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return FloorplanRegion(mask=np.zeros_like(binary, dtype=bool), polygon=[])

    contour = max(contours, key=cv2.contourArea)
    region_mask = np.zeros_like(binary, dtype=np.uint8)
    cv2.drawContours(region_mask, [contour], -1, 1, thickness=cv2.FILLED)

    approx = cv2.approxPolyDP(contour, approx_epsilon, True)
    if len(approx) < 3:
        approx = contour
    polygon = [[int(x), int(y)] for x, y in approx[:, 0]]

    return FloorplanRegion(mask=region_mask.astype(bool), polygon=polygon)


def region_mask_for_image(region_mask: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """Resize a region mask to match image width × height."""
    width, height = image_size
    mask = region_mask > 0.5
    if mask.shape == (height, width):
        return mask
    return cv2.resize(
        mask.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def apply_region_mask(
    image: Image.Image,
    region_mask: np.ndarray,
    *,
    background: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Zero out canvas content outside the floorplan region (white fill)."""
    rgb = np.asarray(image.convert("RGB"))
    mask = region_mask_for_image(region_mask, image.size)
    out = rgb.copy()
    out[~mask] = background
    return Image.fromarray(out, mode="RGB")


def region_crop_box(
    region_mask: np.ndarray,
    image_size: tuple[int, int],
    *,
    padding_px: int = 0,
) -> tuple[int, int, int, int] | None:
    """Axis-aligned crop box ``(x0, y0, x1, y1)`` for a region mask on an image."""
    mask = region_mask_for_image(region_mask, image_size)
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None

    height, width = mask.shape
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    if padding_px > 0:
        y0 = max(0, y0 - padding_px)
        x0 = max(0, x0 - padding_px)
        y1 = min(height, y1 + padding_px)
        x1 = min(width, x1 + padding_px)
    return x0, y0, x1, y1


def crop_region_bbox(
    image: Image.Image,
    region_mask: np.ndarray,
    *,
    padding_px: int = 0,
) -> Image.Image:
    """Crop image to the axis-aligned bounding box of region_mask."""
    box = region_crop_box(region_mask, image.size, padding_px=padding_px)
    if box is None:
        return image.copy()
    return image.crop(box)


