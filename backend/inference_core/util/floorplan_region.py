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


def apply_region_mask(
    image: Image.Image,
    region_mask: np.ndarray,
    *,
    background: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Zero out canvas content outside the floorplan region (white fill)."""
    rgb = np.asarray(image.convert("RGB"))
    mask = region_mask > 0.5
    if mask.shape[:2] != rgb.shape[:2]:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (rgb.shape[1], rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    out = rgb.copy()
    out[~mask] = background
    return Image.fromarray(out, mode="RGB")


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
