"""Snap nearly-coincident room polygon vertices to shared cluster centers."""

from __future__ import annotations

from typing import Any

import numpy as np

_DEFAULT_MIN_CENTER_DISTANCE = 5.0


def _manhattan(a: np.ndarray, b: np.ndarray) -> float:
    return float(abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1])))


def _cluster_centers(points: np.ndarray, min_distance: float) -> list[np.ndarray]:
    order = np.argsort(np.abs(points).sum(axis=1))
    centers: list[np.ndarray] = []
    for idx in order:
        point = points[idx]
        if not centers:
            centers.append(point.copy())
            continue
        if min(_manhattan(point, center) for center in centers) < min_distance:
            continue
        centers.append(point.copy())
    return centers


def _nearest_center(point: np.ndarray, centers: list[np.ndarray]) -> np.ndarray:
    distances = [_manhattan(point, center) for center in centers]
    return centers[int(np.argmin(distances))]


def _dedupe_consecutive(polygon: list[list[int]]) -> list[list[int]]:
    if not polygon:
        return polygon
    deduped = [polygon[0]]
    for point in polygon[1:]:
        if point != deduped[-1]:
            deduped.append(point)
    if len(deduped) > 1 and deduped[0] == deduped[-1]:
        deduped.pop()
    return deduped


def _room_center(polygon: list[list[int]]) -> list[int]:
    cx = sum(point[0] for point in polygon) // len(polygon)
    cy = sum(point[1] for point in polygon) // len(polygon)
    return [cx, cy]


def snap_room_points(
    rooms: list[dict[str, Any]],
    *,
    min_center_distance: float = _DEFAULT_MIN_CENTER_DISTANCE,
) -> list[dict[str, Any]]:
    unique_points: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for room in rooms:
        for x, y in room.get("polygon") or []:
            key = (int(x), int(y))
            if key not in seen:
                seen.add(key)
                unique_points.append(key)

    if not unique_points:
        return [{**room} for room in rooms]

    points = np.array(unique_points, dtype=np.int32)
    centers = _cluster_centers(points, min_center_distance)
    snap_map: dict[tuple[int, int], list[int]] = {}
    for point in points:
        snapped = _nearest_center(point, centers)
        snap_map[(int(point[0]), int(point[1]))] = [int(snapped[0]), int(snapped[1])]

    snapped_rooms: list[dict[str, Any]] = []
    for room in rooms:
        polygon = room.get("polygon") or []
        if not polygon:
            snapped_rooms.append({**room})
            continue
        new_polygon = _dedupe_consecutive(
            [
                snap_map.get((int(x), int(y)), [int(x), int(y)])
                for x, y in polygon
            ]
        )
        updated = {**room, "polygon": new_polygon}
        if len(new_polygon) >= 3:
            updated["center"] = _room_center(new_polygon)
        snapped_rooms.append(updated)
    return snapped_rooms
