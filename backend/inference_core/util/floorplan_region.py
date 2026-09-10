"""Derive a floorplan region mask and boundary polygon from FPN masks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from util.render import draw_mask, draw_polygon_outline, merge_panels


@dataclass(frozen=True)
class FloorplanRegion:
    mask: np.ndarray
    polygon: list[list[int]]


def _dilate_wall_mask(wall_mask: np.ndarray, wall_dilate_px: int) -> np.ndarray:
    binary = wall_mask > 0.5
    if wall_dilate_px <= 0:
        return binary
    k = 2 * wall_dilate_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(binary.astype(np.uint8), kernel).astype(bool)


def _combine_masks(
    room_mask: np.ndarray,
    structure_mask: np.ndarray,
    *,
    wall_dilate_px: int,
) -> np.ndarray:
    height, width = structure_mask.shape[-2:]
    combined = np.zeros((height, width), dtype=bool)
    if room_mask.ndim == 3:
        combined |= room_mask[0] > 0.5
    else:
        combined |= room_mask > 0.5
    if structure_mask.shape[0] > 0:
        combined |= _dilate_wall_mask(structure_mask[0], wall_dilate_px)
    for channel in range(1, min(3, structure_mask.shape[0])):
        combined |= structure_mask[channel] > 0.5
    return combined


def compute_floorplan_region(
    room_mask: np.ndarray,
    structure_mask: np.ndarray,
    *,
    wall_dilate_px: int,
    close_px: int,
    approx_epsilon: float,
) -> FloorplanRegion:
    """OR room/wall/door/window masks, close holes, keep largest patch, fit polygon."""
    binary = _combine_masks(room_mask, structure_mask, wall_dilate_px=wall_dilate_px).astype(np.uint8)
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
    prepared: Image.Image,
    region: FloorplanRegion,
    out_dir: Path,
    *,
    size: int,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    input_view = prepared.convert("RGB")
    if input_view.size != (size, size):
        input_view = input_view.resize((size, size), Image.Resampling.LANCZOS)

    mask = region.mask
    if mask.shape != (size, size):
        mask = cv2.resize(mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST).astype(bool)

    polygon_view = draw_mask(input_view, mask, alpha=120, color=(220, 50, 50))
    if len(region.polygon) >= 3:
        polygon_view = draw_polygon_outline(polygon_view, region.polygon, width=2)
    mask_rgb = np.zeros((size, size, 3), dtype=np.uint8)
    mask_rgb[mask] = (0, 180, 60)
    mask_view = Image.fromarray(mask_rgb, mode="RGB")

    panel_path = out_dir / "floorplan-region.jpg"
    mask_path = out_dir / "floorplan-region-mask.jpg"
    json_path = out_dir / "floorplan-region.json"
    merge_panels([input_view, polygon_view, mask_view]).save(panel_path, format="JPEG", quality=95)
    mask_view.save(mask_path, format="JPEG", quality=95)
    json_path.write_text(
        json.dumps({"size": size, "polygon": region.polygon}, indent=2),
        encoding="utf-8",
    )
    return {
        "panel": panel_path,
        "mask": mask_path,
        "json": json_path,
    }
