"""Wall-subtract room segmentation from FPN structure masks."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from util.render import draw_masks, floorplan_view, merge_panels
from util.room_components import component_masks_from_binary, scale_room_mask


def _wall_at_size(structure_mask: np.ndarray, target_size: int) -> np.ndarray:
    if structure_mask.ndim != 3 or structure_mask.shape[0] == 0:
        raise ValueError(f"structure_mask must be (C, H, W) with C >= 1, got {structure_mask.shape}")

    wall = structure_mask[0].astype(np.float32)
    if wall.shape != (target_size, target_size):
        wall = cv2.resize(wall, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return wall


def _wall_mask_image(wall: np.ndarray) -> Image.Image:
    return ImageOps.invert(Image.fromarray((np.clip(wall, 0.0, 1.0) * 255).astype(np.uint8), mode="L"))


def _dilate_mask(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return np.asarray(mask, dtype=bool)
    k = 2 * px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(np.asarray(mask, dtype=bool).astype(np.uint8), kernel).astype(bool)


def _region_at_size(region_mask: np.ndarray, target_size: int) -> np.ndarray:
    region = region_mask > 0.5
    if region.shape == (target_size, target_size):
        return region
    return cv2.resize(
        region.astype(np.uint8),
        (target_size, target_size),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def _filter_components_by_fpn512(
    wall_components: list[np.ndarray],
    fpn512_room_mask: np.ndarray,
    *,
    size: int,
    overlap_threshold: float = 0.4,
) -> list[np.ndarray]:
    fpn_binary = scale_room_mask(fpn512_room_mask, size) > 0.5
    fpn_components = component_masks_from_binary(fpn_binary.astype(np.uint8))
    if not fpn_components:
        return []

    kept: list[np.ndarray] = []
    for wall_component in wall_components:
        wall_area = int(wall_component.sum())
        if wall_area == 0:
            continue

        best_intersection = 0
        for fpn_component in fpn_components:
            intersection = int(np.logical_and(wall_component, fpn_component).sum())
            if intersection > best_intersection:
                best_intersection = intersection

        if best_intersection == 0:
            continue
        if best_intersection / wall_area <= overlap_threshold:
            continue
        kept.append(wall_component)

    return kept


def _room_mask_from_walls(
    walls: np.ndarray,
    region_mask: np.ndarray,
    fpn512_room_mask: np.ndarray,
    *,
    size: int,
    dilate_px: int = 3,
    overlap_threshold: float = 0.4,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Rooms = region minus dilated walls, filtered by FPN 512px overlap."""
    dilated_walls = _dilate_mask(walls > 0.5, dilate_px)
    region = _region_at_size(region_mask, size)
    rooms = (region & ~dilated_walls).astype(np.uint8)

    wall_components = component_masks_from_binary(rooms)
    kept_components = _filter_components_by_fpn512(
        wall_components,
        fpn512_room_mask,
        size=size,
        overlap_threshold=overlap_threshold,
    )

    result = np.zeros((size, size), dtype=bool)
    for component in kept_components:
        result |= component
    return result.astype(np.float32), kept_components


def render_wall_room_masks(
    structure_mask: np.ndarray,
    fpn512_room_mask: np.ndarray,
    *,
    floorplan: Image.Image,
    region_mask: np.ndarray,
    size: int,
    out_dir: Path,
    step_dir: Path,
) -> dict[str, Path | np.ndarray | list[np.ndarray]]:
    step_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    wall = _wall_at_size(structure_mask, size)
    wall_input = _wall_mask_image(wall)
    input_path = step_dir / f"wall-input-{size}.jpg"
    wall_input.save(input_path, format="JPEG", quality=95)

    wall_subtracted_mask, kept_components = _room_mask_from_walls(
        wall,
        region_mask,
        fpn512_room_mask,
        size=size,
    )

    blank = Image.new("RGB", (size, size), (0, 0, 0))
    floorplan_rgb = floorplan_view(floorplan, size)
    masks_only = draw_masks(blank, kept_components, alpha=255) if kept_components else blank
    overlay = draw_masks(floorplan_rgb, kept_components, alpha=120) if kept_components else floorplan_rgb
    panel = merge_panels([wall_input, overlay, masks_only])

    step_path = step_dir / f"wall-based-room-mask-final-{size}.jpg"
    root_path = out_dir / "wall-based-room-mask-final.jpg"
    panel.save(step_path, format="JPEG", quality=95)
    panel.save(root_path, format="JPEG", quality=95)

    return {
        "input": input_path,
        "wall_based_room_mask_final": step_path,
        "wall_based_room_mask_final_root": root_path,
        "wall_subtracted_mask": wall_subtracted_mask,
        "wall_subtracted_components": kept_components,
    }
