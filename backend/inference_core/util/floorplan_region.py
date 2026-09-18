"""Derive a floorplan region mask and boundary polygon from fpn_five background."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image



@dataclass(frozen=True)
class FloorplanRegion:
    mask: np.ndarray
    polygon: list[list[int]]


def compute_floorplan_region(
    background: np.ndarray,
    *,
    close_px: int,
    approx_epsilon: float,
) -> FloorplanRegion:
    """ROI = invert background; close holes, keep largest patch, fit polygon."""
    plane = background[0] if background.ndim == 3 else background
    binary = (~(plane > 0.5)).astype(np.uint8)
    if not binary.any():
        return FloorplanRegion(
            mask=np.zeros_like(binary, dtype=bool),
            polygon=[],
        )

    if close_px > 0:
        k = 2 * close_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

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


def save_masked_canvas(image: Image.Image, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "input-masked.jpg"
    image.save(path, format="JPEG", quality=95)
    return path


def save_floorplan_region(
    region: FloorplanRegion,
    out_dir: Path,
    *,
    size: int,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    mask = region.mask
    if mask.shape != (size, size):
        mask = cv2.resize(mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST).astype(bool)

    mask_rgb = np.zeros((size, size, 3), dtype=np.uint8)
    mask_rgb[mask] = (0, 180, 60)
    mask_view = Image.fromarray(mask_rgb, mode="RGB")

    mask_path = out_dir / "floorplan-region-mask.jpg"
    json_path = out_dir / "floorplan-region.json"
    mask_view.save(mask_path, format="JPEG", quality=95)
    json_path.write_text(
        json.dumps({"size": size, "polygon": region.polygon}, indent=2),
        encoding="utf-8",
    )
    return {
        "mask": mask_path,
        "json": json_path,
    }
