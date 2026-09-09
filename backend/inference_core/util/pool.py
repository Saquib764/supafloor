"""Grid point pool management and cross-white filtering."""

from __future__ import annotations

import cv2
import numpy as np


def point_filter_gray(gray: np.ndarray, *, dilate_px: int) -> np.ndarray:
    """Invert, dilate walls, invert — expands dark wall bands into room interiors."""
    inverted = (255 - np.asarray(gray, dtype=np.uint8))
    if dilate_px > 0:
        k = 2 * dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        inverted = cv2.dilate(inverted, kernel)
    return (255 - inverted).astype(np.uint8)


def point_in_polygon(x: float, y: float, ring: list[tuple[float, float]]) -> bool:
    inside = False
    j = len(ring) - 1
    for i, (xi, yi) in enumerate(ring):
        xj, yj = ring[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def grid_points_inside_polygon(
    grid_points: list[tuple[float, float]],
    polygon: list[list[int]],
) -> list[tuple[float, float]]:
    ring = [(float(px), float(py)) for px, py in polygon]
    return [point for point in grid_points if point_in_polygon(point[0], point[1], ring)]


def remove_from_grid_pool(
    grid_points: list[tuple[float, float]],
    selected: list[tuple[float, float]],
) -> None:
    selected_set = set(selected)
    grid_points[:] = [point for point in grid_points if point not in selected_set]


def remove_grid_points_inside_polygon(
    grid_points: list[tuple[float, float]],
    polygon: list[list[int]],
) -> int:
    inside = grid_points_inside_polygon(grid_points, polygon)
    if inside:
        remove_from_grid_pool(grid_points, inside)
    return len(inside)


def grid_points_inside_mask(
    grid_points: list[tuple[float, float]],
    mask: np.ndarray,
) -> list[tuple[float, float]]:
    active = np.asarray(mask, dtype=bool)
    height, width = active.shape
    return [
        point
        for point in grid_points
        if 0 <= int(point[0]) < width
        and 0 <= int(point[1]) < height
        and active[int(point[1]), int(point[0])]
    ]


def remove_grid_points_inside_mask(
    grid_points: list[tuple[float, float]],
    mask: np.ndarray,
) -> int:
    inside = grid_points_inside_mask(grid_points, mask)
    if inside:
        remove_from_grid_pool(grid_points, inside)
    return len(inside)


def _arm_is_full_white(arm: np.ndarray) -> bool:
    return arm.size > 0 and float(arm.mean()) >= 250.0


def _cross_arm_full_white_count(gray: np.ndarray, px: int, py: int) -> int:
    height, width = gray.shape
    px = min(max(px, 0), width - 1)
    py = min(max(py, 0), height - 1)
    arms = (
        gray[py, :px],
        gray[py, px:],
        gray[:py, px],
        gray[py:, px],
    )
    return sum(1 for arm in arms if _arm_is_full_white(arm))


def drop_cross_white_canvas_points(
    gray: np.ndarray,
    points: list[tuple[float, float]],
    *,
    min_brightness: int = 127,
) -> list[tuple[float, float]]:
    kept: list[tuple[float, float]] = []
    height, width = gray.shape
    for x, y in points:
        px = int(x)
        py = int(y)
        if not (0 <= px < width and 0 <= py < height):
            continue
        if int(gray[py, px]) < min_brightness:
            continue
        if _cross_arm_full_white_count(gray, px, py) >= 2:
            continue
        kept.append((x, y))
    return kept
