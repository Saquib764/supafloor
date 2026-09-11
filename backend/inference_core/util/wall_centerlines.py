"""Wall centerline skeletons and polyline fitting from the wall-subtract wall mask."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize

from util.render import draw_mask, draw_polylines, floorplan_view, merge_panels, with_label
from util.room_components import scale_room_mask

Coord = tuple[int, int]
PathCoords = list[Coord]

OPENING_COLORS = {
    "doors": (220, 50, 50),
    "windows": (50, 120, 220),
}

OpeningCenterlines = dict[str, dict[str, np.ndarray | list[list[list[int]]]]]


def wall_binary_from_masks(wall_masks: list[np.ndarray], size: int) -> np.ndarray:
    wall = np.zeros((size, size), dtype=bool)
    for wall_mask in wall_masks:
        wall |= scale_room_mask(wall_mask, size) > 0.5
    return wall


def opencv_thinning(binary: np.ndarray) -> np.ndarray:
    img = binary.astype(np.uint8) * 255
    thinned = cv2.ximgproc.thinning(img, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    return thinned > 0


def skimage_skeleton(binary: np.ndarray) -> np.ndarray:
    return skeletonize(binary)


def wall_skeletons_from_masks(
    wall_masks: list[np.ndarray],
    *,
    size: int,
) -> dict[str, np.ndarray]:
    binary = wall_binary_from_masks(wall_masks, size)
    return {
        "wall_mask": binary,
        "opencv_thinning": opencv_thinning(binary),
        "skimage_skeleton": skimage_skeleton(binary),
    }


def _neighbors8(y: int, x: int, skel: np.ndarray) -> list[Coord]:
    height, width = skel.shape
    neighbors: list[Coord] = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width and skel[ny, nx]:
                neighbors.append((ny, nx))
    return neighbors


def _degree_map(skel: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    counts = cv2.filter2D(skel.astype(np.uint8), cv2.CV_16S, kernel, borderType=cv2.BORDER_CONSTANT)
    return counts.astype(np.int16)


def _path_length(path: PathCoords) -> float:
    if len(path) < 2:
        return 0.0
    total = 0.0
    for (y0, x0), (y1, x1) in zip(path[:-1], path[1:]):
        total += float(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5)
    return total


def _prune_spurs(skel: np.ndarray, max_spur_len: int) -> np.ndarray:
    if max_spur_len <= 0:
        return skel.copy()

    out = skel.copy()
    changed = True
    while changed:
        changed = False
        degree = _degree_map(out)
        for y, x in np.argwhere(out & (degree == 1)):
            path: PathCoords = [(int(y), int(x))]
            prev: Coord | None = None
            cur = (int(y), int(x))
            terminated_at_junction = False

            while len(path) <= max_spur_len:
                nbrs = [n for n in _neighbors8(*cur, out) if n != prev]
                if len(nbrs) != 1:
                    terminated_at_junction = len(nbrs) >= 2
                    break
                nxt = nbrs[0]
                path.append(nxt)
                prev, cur = cur, nxt

            if terminated_at_junction and len(path) <= max_spur_len:
                for py, px in path:
                    out[py, px] = False
                changed = True

    return out


def _trace_chain(
    start: Coord,
    nbr: Coord,
    skel: np.ndarray,
    visited_edges: set[tuple[Coord, Coord]],
) -> PathCoords:
    path = [start, nbr]
    edge = tuple(sorted((start, nbr)))
    visited_edges.add(edge)

    prev = start
    cur = nbr
    while True:
        nbrs = [n for n in _neighbors8(*cur, skel) if n != prev]
        if len(nbrs) != 1:
            break
        nxt = nbrs[0]
        edge = tuple(sorted((cur, nxt)))
        if edge in visited_edges:
            break
        visited_edges.add(edge)
        path.append(nxt)
        prev, cur = cur, nxt
    return path


def _trace_loop(
    start: Coord,
    first: Coord,
    skel: np.ndarray,
    visited_edges: set[tuple[Coord, Coord]],
) -> PathCoords:
    path = [start, first]
    edge = tuple(sorted((start, first)))
    visited_edges.add(edge)

    prev = start
    cur = first
    while cur != start:
        nbrs = [n for n in _neighbors8(*cur, skel) if n != prev]
        if len(nbrs) != 1:
            break
        nxt = nbrs[0]
        edge = tuple(sorted((cur, nxt)))
        if edge in visited_edges:
            break
        visited_edges.add(edge)
        path.append(nxt)
        prev, cur = cur, nxt
    return path


def _simplify_path(path: PathCoords, epsilon: float) -> list[list[int]]:
    if len(path) < 2:
        return []
    points = np.array([[x, y] for y, x in path], dtype=np.float32).reshape(-1, 1, 2)
    simplified = cv2.approxPolyDP(points, epsilon, closed=False)
    if len(simplified) < 2:
        return [[int(x), int(y)] for y, x in path]
    return [[int(x), int(y)] for x, y in simplified[:, 0]]


def _polyline_length(polyline: list[list[int]]) -> float:
    if len(polyline) < 2:
        return 0.0
    total = 0.0
    for (x0, y0), (x1, y1) in zip(polyline[:-1], polyline[1:]):
        total += float(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5)
    return total


def _wall_segments(wall_polylines: list[list[list[int]]]) -> list[tuple[list[int], list[int]]]:
    segments: list[tuple[list[int], list[int]]] = []
    for polyline in wall_polylines:
        if len(polyline) < 2:
            continue
        for start, end in zip(polyline[:-1], polyline[1:]):
            segments.append((start, end))
    return segments


def _indexed_wall_segments(
    wall_polylines: list[list[list[int]]],
) -> list[tuple[int, list[int], list[int]]]:
    segments: list[tuple[int, list[int], list[int]]] = []
    for poly_idx, polyline in enumerate(wall_polylines):
        if len(polyline) < 2:
            continue
        for start, end in zip(polyline[:-1], polyline[1:]):
            segments.append((poly_idx, start, end))
    return segments


def _point_to_segment_distance(
    px: float,
    py: float,
    start: list[int],
    end: list[int],
) -> float:
    ax, ay = float(start[0]), float(start[1])
    bx, by = float(end[0]), float(end[1])
    dx, dy = bx - ax, by - ay
    len2 = dx * dx + dy * dy
    if len2 < 1e-12:
        return float(((px - ax) ** 2 + (py - ay) ** 2) ** 0.5)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / len2))
    qx = ax + t * dx
    qy = ay + t * dy
    return float(((px - qx) ** 2 + (py - qy) ** 2) ** 0.5)


def _nearest_segment_distance(
    point: list[int],
    segments: list[tuple[list[int], list[int]]],
) -> float:
    px, py = float(point[0]), float(point[1])
    best = float("inf")
    for start, end in segments:
        best = min(best, _point_to_segment_distance(px, py, start, end))
    return best


def _nearest_wall_segment_distance(
    point: list[int],
    segments: list[tuple[int, list[int], list[int]]],
    exclude_poly_idx: int,
) -> float:
    px, py = float(point[0]), float(point[1])
    best = float("inf")
    for poly_idx, start, end in segments:
        if poly_idx == exclude_poly_idx:
            continue
        best = min(best, _point_to_segment_distance(px, py, start, end))
    return best


def filter_wall_polylines_by_endpoint_proximity(
    polylines: list[list[list[int]]],
    *,
    max_endpoint_dist: float = 20.0,
) -> list[list[list[int]]]:
    """Drop wall polylines whose start or end is far from every other wall segment."""
    segments = _indexed_wall_segments(polylines)
    if not segments:
        return []

    kept: list[list[list[int]]] = []
    for poly_idx, polyline in enumerate(polylines):
        if len(polyline) < 2:
            continue
        start_dist = _nearest_wall_segment_distance(polyline[0], segments, poly_idx)
        end_dist = _nearest_wall_segment_distance(polyline[-1], segments, poly_idx)
        if start_dist > max_endpoint_dist or end_dist > max_endpoint_dist:
            continue
        kept.append(polyline)
    return kept


def filter_opening_polylines_by_wall_proximity(
    polylines: list[list[list[int]]],
    wall_polylines: list[list[list[int]]],
    *,
    max_endpoint_dist: float = 20.0,
) -> list[list[list[int]]]:
    """Drop opening polylines whose start or end is far from every wall segment."""
    wall_segments = _wall_segments(wall_polylines)
    if not wall_segments:
        return []

    kept: list[list[list[int]]] = []
    for polyline in polylines:
        if len(polyline) < 2:
            continue
        start_dist = _nearest_segment_distance(polyline[0], wall_segments)
        end_dist = _nearest_segment_distance(polyline[-1], wall_segments)
        if start_dist > max_endpoint_dist or end_dist > max_endpoint_dist:
            continue
        kept.append(polyline)
    return kept


def filter_openings_to_walls(
    opening_centerlines: OpeningCenterlines,
    wall_polylines: list[list[list[int]]],
    *,
    max_endpoint_dist: float = 20.0,
) -> OpeningCenterlines:
    filtered: OpeningCenterlines = {}
    for key, entry in opening_centerlines.items():
        polylines = entry["polylines"]  # type: ignore[assignment]
        filtered[key] = {
            "mask": entry["mask"],  # type: ignore[typeddict-item]
            "skeleton": entry["skeleton"],  # type: ignore[typeddict-item]
            "polylines": filter_opening_polylines_by_wall_proximity(
                polylines,
                wall_polylines,
                max_endpoint_dist=max_endpoint_dist,
            ),
        }
    return filtered


def _project_point_to_line(
    point: tuple[float, float],
    start: list[int],
    end: list[int],
) -> tuple[tuple[float, float], float]:
    px, py = point
    ax, ay = float(start[0]), float(start[1])
    bx, by = float(end[0]), float(end[1])
    dx, dy = bx - ax, by - ay
    len2 = dx * dx + dy * dy
    if len2 < 1e-12:
        return (ax, ay), float(((px - ax) ** 2 + (py - ay) ** 2) ** 0.5)
    t = ((px - ax) * dx + (py - ay) * dy) / len2
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    dist = float(((px - proj_x) ** 2 + (py - proj_y) ** 2) ** 0.5)
    return (proj_x, proj_y), dist


def _snap_polyline_to_walls(
    polyline: list[list[int]],
    wall_segments: list[tuple[list[int], list[int]]],
) -> list[list[int]]:
    if len(polyline) < 2 or not wall_segments:
        return polyline

    start = polyline[0]
    end = polyline[-1]
    length = float(((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5)
    if length < 1e-6:
        return polyline

    midpoint = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)

    best_segment: tuple[list[int], list[int]] | None = None
    best_proj: tuple[float, float] | None = None
    best_dist = float("inf")
    for seg_start, seg_end in wall_segments:
        dist = _point_to_segment_distance(midpoint[0], midpoint[1], seg_start, seg_end)
        if dist >= best_dist:
            continue
        proj, _ = _project_point_to_line(midpoint, seg_start, seg_end)
        best_dist = dist
        best_segment = (seg_start, seg_end)
        best_proj = proj

    if best_segment is None or best_proj is None:
        return polyline

    seg_start, seg_end = best_segment
    dx = float(seg_end[0] - seg_start[0])
    dy = float(seg_end[1] - seg_start[1])
    seg_len = (dx * dx + dy * dy) ** 0.5
    if seg_len < 1e-6:
        return polyline

    ux, uy = dx / seg_len, dy / seg_len
    half = length / 2.0
    proj_x, proj_y = best_proj
    snapped_start = [int(round(proj_x - ux * half)), int(round(proj_y - uy * half))]
    snapped_end = [int(round(proj_x + ux * half)), int(round(proj_y + uy * half))]
    return [snapped_start, snapped_end]


def snap_openings_to_walls(
    opening_centerlines: OpeningCenterlines,
    wall_polylines: list[list[list[int]]],
) -> OpeningCenterlines:
    wall_segments = _wall_segments(wall_polylines)
    if not wall_segments:
        return opening_centerlines

    snapped: OpeningCenterlines = {}
    for key, entry in opening_centerlines.items():
        polylines = entry["polylines"]  # type: ignore[assignment]
        snapped_polylines = [
            _snap_polyline_to_walls(polyline, wall_segments)
            for polyline in polylines
            if len(polyline) >= 2
        ]
        snapped[key] = {
            "mask": entry["mask"],  # type: ignore[typeddict-item]
            "skeleton": entry["skeleton"],  # type: ignore[typeddict-item]
            "polylines": snapped_polylines,
        }
    return snapped


def polylines_from_skeleton(
    skeleton: np.ndarray,
    *,
    epsilon: float = 3.0,
    min_length: float = 12.0,
    max_spur_len: int = 8,
) -> list[list[list[int]]]:
    """Trace skeleton graph and simplify each branch with Douglas-Peucker."""
    skel = _prune_spurs(np.asarray(skeleton, dtype=bool), max_spur_len)
    if not skel.any():
        return []

    degree = _degree_map(skel)
    keypoints = {
        (int(y), int(x))
        for y, x in np.argwhere(skel)
        if degree[y, x] != 2
    }

    visited_edges: set[tuple[Coord, Coord]] = set()
    raw_paths: list[PathCoords] = []

    for start in sorted(keypoints):
        for nbr in _neighbors8(*start, skel):
            edge = tuple(sorted((start, nbr)))
            if edge in visited_edges:
                continue
            raw_paths.append(_trace_chain(start, nbr, skel, visited_edges))

    for y, x in np.argwhere(skel):
        start = (int(y), int(x))
        for nbr in _neighbors8(*start, skel):
            edge = tuple(sorted((start, nbr)))
            if edge in visited_edges:
                continue
            raw_paths.append(_trace_loop(start, nbr, skel, visited_edges))
            break

    polylines: list[list[list[int]]] = []
    for path in raw_paths:
        if _path_length(path) < min_length:
            continue
        polyline = _simplify_path(path, epsilon)
        if _polyline_length(polyline) < min_length:
            continue
        polylines.append(polyline)

    return polylines


def _polyline_stroke(size: int) -> int:
    return max(6, size // 170)


def _opening_polyline_stroke(size: int) -> int:
    return max(10, size // 90)


def _overlay_opening_masks(
    image: Image.Image,
    opening_centerlines: OpeningCenterlines | None,
    *,
    alpha: int,
) -> Image.Image:
    if not opening_centerlines:
        return image
    overlay = image
    for key, color in OPENING_COLORS.items():
        entry = opening_centerlines.get(key)
        if not entry:
            continue
        overlay = draw_mask(overlay, entry["mask"], alpha=alpha, color=color)  # type: ignore[arg-type]
    return overlay


def _draw_structure_polylines(
    image: Image.Image,
    wall_polylines: list[list[list[int]]],
    *,
    wall_color: tuple[int, int, int],
    opening_centerlines: OpeningCenterlines | None,
    stroke: int,
) -> Image.Image:
    overlay = draw_polylines(image, wall_polylines, color=wall_color, width=stroke)
    if not opening_centerlines:
        return overlay
    opening_stroke = _opening_polyline_stroke(image.size[0])
    for key, color in OPENING_COLORS.items():
        entry = opening_centerlines.get(key)
        if not entry:
            continue
        overlay = draw_polylines(overlay, entry["polylines"], color=color, width=opening_stroke)  # type: ignore[arg-type]
    return overlay


def _structure_centerline_image(
    floorplan: Image.Image,
    wall_polylines: list[list[list[int]]],
    *,
    size: int,
    opening_centerlines: OpeningCenterlines | None = None,
) -> Image.Image:
    base = floorplan_view(floorplan, size).convert("RGB")
    return _draw_structure_polylines(
        base,
        wall_polylines,
        wall_color=(0, 100, 0),
        opening_centerlines=opening_centerlines,
        stroke=_polyline_stroke(size),
    )


def skeleton_polyline_overlay_row(
    floorplan: Image.Image,
    mask: np.ndarray,
    polylines: list[list[list[int]]],
    unfiltered_polylines: list[list[list[int]]],
    *,
    size: int,
    label: str,
    color: tuple[int, int, int],
    opening_centerlines: OpeningCenterlines | None = None,
) -> Image.Image:
    view = floorplan_view(floorplan, size)
    blank = Image.new("RGB", (size, size), (255, 255, 255))
    stroke = _polyline_stroke(size)
    on_view = _overlay_opening_masks(draw_mask(view, mask, alpha=120, color=color), opening_centerlines, alpha=120)
    on_blank = _overlay_opening_masks(draw_mask(blank, mask, alpha=255, color=color), opening_centerlines, alpha=255)
    return merge_panels(
        [
            view,
            with_label(on_view, label),
            with_label(on_blank, label),
            with_label(
                draw_polylines(blank, unfiltered_polylines, color=color, width=stroke),
                "Unfiltered polylines",
            ),
            with_label(
                _draw_structure_polylines(
                    blank,
                    polylines,
                    wall_color=color,
                    opening_centerlines=opening_centerlines,
                    stroke=stroke,
                ),
                "Filtered polylines",
            ),
        ]
    )


def save_wall_centerline_outputs(
    floorplan: Image.Image,
    skeletons: dict[str, np.ndarray],
    out_dir: Path,
    *,
    size: int,
    opening_centerlines: OpeningCenterlines | None = None,
    polyline_epsilon: float = 3.0,
    polyline_min_length: float = 12.0,
    polyline_max_spur_len: int = 8,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    opencv_polylines_unfiltered = polylines_from_skeleton(
        skeletons["opencv_thinning"],
        epsilon=polyline_epsilon,
        min_length=polyline_min_length,
        max_spur_len=polyline_max_spur_len,
    )
    skimage_polylines_unfiltered = polylines_from_skeleton(
        skeletons["skimage_skeleton"],
        epsilon=polyline_epsilon,
        min_length=polyline_min_length,
        max_spur_len=polyline_max_spur_len,
    )
    opencv_polylines = filter_wall_polylines_by_endpoint_proximity(opencv_polylines_unfiltered)
    skimage_polylines = filter_wall_polylines_by_endpoint_proximity(skimage_polylines_unfiltered)

    if opening_centerlines:
        opening_centerlines = filter_openings_to_walls(opening_centerlines, skimage_polylines)
        opening_centerlines = snap_openings_to_walls(opening_centerlines, skimage_polylines)

    wall_centerline_path = out_dir / "wall-centerline.jpg"
    _structure_centerline_image(
        floorplan,
        skimage_polylines,
        size=size,
        opening_centerlines=opening_centerlines,
    ).save(wall_centerline_path, format="JPEG", quality=95)
    outputs["wall_centerline"] = wall_centerline_path

    opencv_path = out_dir / "wall-centerline-opencv-thinning.jpg"
    skeleton_polyline_overlay_row(
        floorplan,
        skeletons["opencv_thinning"],
        opencv_polylines,
        opencv_polylines_unfiltered,
        size=size,
        label="OpenCV thinning",
        color=(40, 40, 40),
        opening_centerlines=opening_centerlines,
    ).save(opencv_path, format="JPEG", quality=95)
    outputs["opencv_thinning"] = opencv_path

    skimage_path = out_dir / "wall-centerline-skimage-skeleton.jpg"
    skeleton_polyline_overlay_row(
        floorplan,
        skeletons["skimage_skeleton"],
        skimage_polylines,
        skimage_polylines_unfiltered,
        size=size,
        label="skimage skeletonize",
        color=(40, 40, 40),
        opening_centerlines=opening_centerlines,
    ).save(skimage_path, format="JPEG", quality=95)
    outputs["skimage_skeleton"] = skimage_path

    polylines_path = out_dir / "wall-centerline-polylines.json"
    payload: dict[str, object] = {
        "size": size,
        "opencv_thinning": opencv_polylines,
        "skimage_skeleton": skimage_polylines,
    }
    if opening_centerlines:
        payload["doors"] = opening_centerlines["doors"]["polylines"]
        payload["windows"] = opening_centerlines["windows"]["polylines"]
    polylines_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    outputs["polylines_json"] = polylines_path

    return outputs
