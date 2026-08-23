"""Insert T-joint vertices where another room's wall meets a wall interior."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

_MAX_T_JOINT_DISTANCE = 5.0
_MIN_T_JOINT_ANGLE_DEG = 15.0
_MIN_ENDPOINT_DISTANCE = 1.0


def _as_point(point: list[int] | tuple[int, int]) -> np.ndarray:
    return np.array([int(point[0]), int(point[1])], dtype=np.float64)


def _point_segment_distance_t(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> tuple[float, float]:
    edge = end - start
    length_sq = float(np.dot(edge, edge))
    if length_sq == 0.0:
        return float("inf"), 0.0
    t = float(np.dot(point - start, edge) / length_sq)
    projection = start + t * edge
    return float(np.linalg.norm(point - projection)), t


def _acute_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    cosine = float(np.dot(a, b) / (norm_a * norm_b))
    cosine = max(-1.0, min(1.0, cosine))
    angle = math.degrees(math.acos(cosine))
    return min(angle, 180.0 - angle)


def _interior_projection(
    start: np.ndarray,
    end: np.ndarray,
    t: float,
) -> bool:
    if t <= 0.0 or t >= 1.0:
        return False
    edge = end - start
    length = float(np.linalg.norm(edge))
    if length == 0.0:
        return False
    dist_start = t * length
    dist_end = (1.0 - t) * length
    return dist_start >= _MIN_ENDPOINT_DISTANCE and dist_end >= _MIN_ENDPOINT_DISTANCE


def _wall_angle_ok(
    wall_start: np.ndarray,
    wall_end: np.ndarray,
    other_polygon: list[list[int]],
    vertex_index: int,
) -> bool:
    vertex_count = len(other_polygon)
    if vertex_count < 2:
        return False
    vertex = _as_point(other_polygon[vertex_index])
    prev_vertex = _as_point(other_polygon[(vertex_index - 1) % vertex_count])
    next_vertex = _as_point(other_polygon[(vertex_index + 1) % vertex_count])
    wall_dir = wall_end - wall_start
    edge_to_prev = vertex - prev_vertex
    edge_to_next = next_vertex - vertex
    return max(
        _acute_angle_deg(wall_dir, edge_to_prev),
        _acute_angle_deg(wall_dir, edge_to_next),
    ) > _MIN_T_JOINT_ANGLE_DEG


def _closest_t_joint(
    wall_start: list[int],
    wall_end: list[int],
    rooms: list[dict[str, Any]],
    room_index: int,
) -> tuple[list[int], float] | None:
    start = _as_point(wall_start)
    end = _as_point(wall_end)
    best_point: list[int] | None = None
    best_distance = float("inf")
    best_t = 0.0

    for other_index, other_room in enumerate(rooms):
        if other_index == room_index:
            continue
        other_polygon = other_room.get("polygon") or []
        for vertex_index, vertex in enumerate(other_polygon):
            if vertex == wall_start or vertex == wall_end:
                continue
            point = _as_point(vertex)
            distance, t = _point_segment_distance_t(point, start, end)
            if not _interior_projection(start, end, t):
                continue
            if distance >= best_distance:
                continue
            if not _wall_angle_ok(start, end, other_polygon, vertex_index):
                continue
            best_point = [int(vertex[0]), int(vertex[1])]
            best_distance = distance
            best_t = t

    if best_point is None or best_distance > _MAX_T_JOINT_DISTANCE:
        return None
    return best_point, best_t


def _apply_edge_insertions(
    polygon: list[list[int]],
    insertions: dict[int, list[tuple[list[int], float]]],
) -> list[list[int]]:
    if not insertions:
        return polygon

    updated = polygon[:]
    for edge_index in sorted(insertions, reverse=True):
        points = sorted(insertions[edge_index], key=lambda item: item[1])
        insert_at = edge_index + 1
        for point, _t in reversed(points):
            if point in updated:
                continue
            updated[insert_at:insert_at] = [point]
    return updated


def _room_center(polygon: list[list[int]]) -> list[int]:
    cx = sum(point[0] for point in polygon) // len(polygon)
    cy = sum(point[1] for point in polygon) // len(polygon)
    return [cx, cy]


def insert_t_joints(rooms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    updated_rooms: list[dict[str, Any]] = []
    working = [{**room, "polygon": list(room.get("polygon") or [])} for room in rooms]

    for room_index, room in enumerate(working):
        polygon = room.get("polygon") or []
        if len(polygon) < 3:
            updated_rooms.append(room)
            continue

        insertions: dict[int, list[tuple[list[int], float]]] = {}
        for edge_index in range(len(polygon)):
            wall_start = polygon[edge_index]
            wall_end = polygon[(edge_index + 1) % len(polygon)]
            match = _closest_t_joint(wall_start, wall_end, working, room_index)
            if match is None:
                continue
            point, t = match
            insertions.setdefault(edge_index, []).append((point, t))

        new_polygon = _apply_edge_insertions(polygon, insertions)
        updated = {**room, "polygon": new_polygon}
        if len(new_polygon) >= 3:
            updated["center"] = _room_center(new_polygon)
        updated_rooms.append(updated)

    return updated_rooms
