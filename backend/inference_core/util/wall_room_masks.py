"""Wall-subtract room segmentation from wall and room masks."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from util.room_components import component_masks_from_binary, scale_room_mask
from util.render import draw_masks

def _mask_at_size(mask: np.ndarray, target_size: int) -> np.ndarray:
    plane = mask[0].astype(np.float32) if mask.ndim == 3 else mask.astype(np.float32)
    if plane.shape != (target_size, target_size):
        plane = cv2.resize(plane, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return plane


def _structure_barrier(
    wall_masks: list[np.ndarray],
    door_mask: np.ndarray,
    window_mask: np.ndarray,
    *,
    size: int,
) -> np.ndarray:
    """Walls (OR'd), door, and window scaled to size, then overlaid."""
    wall = np.zeros((size, size), dtype=bool)
    for wall_mask in wall_masks:
        wall |= _mask_at_size(wall_mask, size) > 0.5
    door = _mask_at_size(door_mask, size) > 0.5
    window = _mask_at_size(window_mask, size) > 0.5
    return (wall | door | window).astype(np.float32)


def _wall_mask_image(structure: np.ndarray) -> Image.Image:
    return ImageOps.invert(Image.fromarray((np.clip(structure, 0.0, 1.0) * 255).astype(np.uint8), mode="L"))


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
    structure: np.ndarray,
    region_mask: np.ndarray,
    fpn512_room_mask: np.ndarray,
    *,
    size: int,
    overlap_threshold: float = 0.4,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Rooms = region minus wall+openings; return (raw components, filtered)."""
    barrier = structure > 0.5
    region = _region_at_size(region_mask, size)
    rooms = (region & ~barrier).astype(np.uint8)

    wall_components = component_masks_from_binary(rooms)
    kept_components = _filter_components_by_fpn512(
        wall_components,
        fpn512_room_mask,
        size=size,
        overlap_threshold=overlap_threshold,
    )
    return wall_components, kept_components


def render_wall_room_masks(
    wall_masks: list[np.ndarray],
    door_mask: np.ndarray,
    window_mask: np.ndarray,
    fpn512_room_mask: np.ndarray,
    *,
    region_mask: np.ndarray,
    size: int,
    step_dir: Path,
) -> dict[str, Path | list[np.ndarray]]:
    step_dir.mkdir(parents=True, exist_ok=True)

    structure = _structure_barrier(wall_masks, door_mask, window_mask, size=size)
    wall_input = _wall_mask_image(structure)
    input_path = step_dir / f"wall-input-{size}.jpg"
    wall_input.save(input_path, format="JPEG", quality=95)

    raw_components, kept_components = _room_mask_from_walls(
        structure,
        region_mask,
        fpn512_room_mask,
        size=size,
    )

    blank = Image.new("RGB", (size, size), (255, 255, 255))
    prefilter = (
        draw_masks(blank, raw_components, alpha=255)
        if raw_components
        else blank
    )
    prefilter_path = step_dir / f"wall-subtract-prefilter-{size}.jpg"
    prefilter.save(prefilter_path, format="JPEG", quality=95)

    return {
        "input": input_path,
        "prefilter": prefilter_path,
        "wall_subtracted_components": kept_components,
    }
