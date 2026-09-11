"""Door and window centerlines from FPN masks."""

from __future__ import annotations

from typing import TypedDict

import numpy as np

from util.room_components import component_masks_from_binary, scale_room_mask
from util.wall_centerlines import opencv_thinning, polylines_from_skeleton


class OpeningCenterlineEntry(TypedDict):
    mask: np.ndarray
    skeleton: np.ndarray
    polylines: list[list[list[int]]]


def _mask_at_size(mask: np.ndarray, target_size: int) -> np.ndarray:
    return scale_room_mask(mask, target_size) > 0.5


def _binary_with_min_area(binary: np.ndarray, min_area_px: int) -> np.ndarray:
    filtered = np.zeros_like(binary, dtype=bool)
    for component in component_masks_from_binary(binary.astype(np.uint8)):
        if int(component.sum()) >= min_area_px:
            filtered |= component.astype(bool)
    return filtered


def opening_centerlines_from_masks(
    door_mask: np.ndarray,
    window_mask: np.ndarray,
    *,
    size: int,
    min_area_px: int,
    polyline_epsilon: float,
    polyline_min_length: float,
    polyline_max_spur_len: int,
) -> dict[str, OpeningCenterlineEntry]:
    openings = {
        "doors": _binary_with_min_area(_mask_at_size(door_mask, size), min_area_px),
        "windows": _binary_with_min_area(_mask_at_size(window_mask, size), min_area_px),
    }
    results: dict[str, OpeningCenterlineEntry] = {}
    for key, binary in openings.items():
        skeleton = opencv_thinning(binary)
        polylines = polylines_from_skeleton(
            skeleton,
            epsilon=polyline_epsilon,
            min_length=polyline_min_length,
            max_spur_len=polyline_max_spur_len,
        )
        results[key] = {
            "mask": binary,
            "skeleton": skeleton,
            "polylines": polylines,
        }
    return results
