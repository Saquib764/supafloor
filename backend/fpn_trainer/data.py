"""Hub rows → grayscale floor plan input + 3-channel structure targets."""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from copy import deepcopy
from typing import Any, Iterator

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from registry import hub_dataset_row_count, load_hub_dataset, load_hub_rows_at
from shapely.geometry import Polygon
from torch.utils.data import IterableDataset

HUB_COLUMNS = ("stem", "image", "simple_image", "extraction")
_EXTRACTION_CANVAS = 1024
ROOM_WALL_THICK = 4.0
BOUNDARY_WALL_THICK = 8.0
_GEOM_TOL = 0.5


def extraction_scale(input_size: int) -> float:
    return input_size / _EXTRACTION_CANVAS


def prepare_image(image: Image.Image, size: int) -> Image.Image:
    """Grayscale → resize → invert (white walls on black background)."""
    gray = image.convert("L")
    if gray.size != (size, size):
        gray = gray.resize((size, size), Image.Resampling.LANCZOS)
    return ImageOps.invert(gray)


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim != 2:
        raise ValueError(f"expected single-channel image, got shape {array.shape}")
    return torch.from_numpy(array).unsqueeze(0)


def _scale_point(point: dict[str, Any], scale: float) -> dict[str, float]:
    return {"x": float(point["x"]) * scale, "y": float(point["y"]) * scale}


def _scale_segment(segment: dict[str, Any], scale: float) -> dict[str, Any]:
    return {
        **segment,
        "start": _scale_point(segment["start"], scale),
        "end": _scale_point(segment["end"], scale),
    }


def scale_extraction(extraction: dict[str, Any], scale: float) -> dict[str, Any]:
    if scale == 1.0:
        return extraction
    scaled = deepcopy(extraction)
    for room in scaled.get("rooms") or []:
        room["polygon"] = [_scale_point(point, scale) for point in room.get("polygon") or []]
    openings = scaled.get("openings") or {}
    scaled["openings"] = {
        "doors": [_scale_segment(door, scale) for door in openings.get("doors") or []],
        "windows": [_scale_segment(window, scale) for window in openings.get("windows") or []],
    }
    scaled["pillars"] = [
        {
            **pillar,
            "x": float(pillar["x"]) * scale,
            "y": float(pillar["y"]) * scale,
            "width": float(pillar["width"]) * scale,
            "height": float(pillar["height"]) * scale,
        }
        for pillar in scaled.get("pillars") or []
    ]
    return scaled


def _openings(extraction: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    openings = extraction.get("openings") or {}
    return list(openings.get("doors") or []), list(openings.get("windows") or [])


def _select_hub_columns(dataset: Any) -> Any:
    missing = [col for col in HUB_COLUMNS if col not in dataset.column_names]
    if missing:
        raise KeyError(f"Hub dataset missing columns: {missing}")
    return dataset.select_columns(list(HUB_COLUMNS))


def load_train_split(
    dataset: str,
    *,
    test_fraction: float,
    revision: str | None = None,
    cache_dir: Any = None,
) -> tuple[Any, int, int]:
    """Load cached Hub train rows by index (no streaming — avoids parquet generator cleanup bugs)."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("data.test_fraction must be between 0 and 1")
    total = hub_dataset_row_count(dataset, revision=revision, cache_dir=cache_dir)
    if total < 2:
        raise ValueError("need at least 2 rows to split into train and test")
    train_pct = int(round((1.0 - test_fraction) * 100))
    if not 0 < train_pct < 100:
        raise ValueError("data.test_fraction must leave at least one train and test row")
    train_count = int(total * train_pct / 100)
    test_count = total - train_count
    rows = _select_hub_columns(
        load_hub_dataset(dataset, revision=revision, cache_dir=cache_dir)
    )
    if len(rows) < total:
        raise ValueError(f"Hub dataset has {len(rows)} rows, expected {total}")
    return rows, train_count, test_count


def _plan_id(stem: str) -> str:
    if "_" not in stem:
        return stem
    plan_id, rest = stem.rsplit("_", 1)
    try:
        int(rest)
    except ValueError:
        return stem
    return plan_id


def load_viz_rows(
    dataset: str,
    train_count: int,
    *,
    count: int,
    seed: int | None = None,
    revision: str | None = None,
    cache_dir: Any = None,
) -> list[dict[str, Any]]:
    """Random test-split rows, one variant per floor plan."""
    if count < 1:
        return []
    total = hub_dataset_row_count(dataset, revision=revision, cache_dir=cache_dir)
    test_indices = list(range(train_count, total))
    if not test_indices:
        return []
    stems = load_hub_rows_at(
        dataset,
        test_indices,
        columns=("stem",),
        revision=revision,
        cache_dir=cache_dir,
    )
    by_plan: dict[str, list[int]] = {}
    for index, row in zip(test_indices, stems, strict=True):
        plan = _plan_id(str(row["stem"]))
        by_plan.setdefault(plan, []).append(index)
    rng = random.Random(seed)
    plans = list(by_plan)
    rng.shuffle(plans)
    picked = plans[: min(count, len(plans))]
    indices = [rng.choice(by_plan[plan]) for plan in picked]
    return load_hub_rows_at(
        dataset,
        indices,
        columns=HUB_COLUMNS,
        revision=revision,
        cache_dir=cache_dir,
    )


def _polygon_from_vertices(vertices: list[dict[str, Any]]) -> Polygon:
    return Polygon([(float(point["x"]), float(point["y"])) for point in vertices])


def _edge_key(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float, float, float]:
    if a > b:
        a, b = b, a
    return (round(a[0], 1), round(a[1], 1), round(b[0], 1), round(b[1], 1))


def _project_t(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    len2 = dx * dx + dy * dy
    if len2 < 1e-12:
        return 0.0
    return ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / len2


def _dist_to_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    t = max(0.0, min(1.0, _project_t(point, start, end)))
    q = (
        start[0] + (end[0] - start[0]) * t,
        start[1] + (end[1] - start[1]) * t,
    )
    dx = point[0] - q[0]
    dy = point[1] - q[1]
    return math.hypot(dx, dy)


def _opening_wall_span(
    opening_start: tuple[float, float],
    opening_end: tuple[float, float],
    wall_start: tuple[float, float],
    wall_end: tuple[float, float],
) -> float | None:
    if _dist_to_segment(opening_start, wall_start, wall_end) > _GEOM_TOL:
        return None
    if _dist_to_segment(opening_end, wall_start, wall_end) > _GEOM_TOL:
        return None
    t0 = _project_t(opening_start, wall_start, wall_end)
    t1 = _project_t(opening_end, wall_start, wall_end)
    lo, hi = sorted((t0, t1))
    if hi < -0.02 or lo > 1.02:
        return None
    return max(0.0, hi - lo)


def _opening_wall_thickness(
    opening_start: tuple[float, float],
    opening_end: tuple[float, float],
    walls: list[dict[str, Any]],
    *,
    room_wall_thick: float,
    boundary_wall_thick: float,
) -> float:
    best_span = 0.0
    thickness = room_wall_thick
    for wall in walls:
        wall_start = _wall_endpoint(wall, "start")
        wall_end = _wall_endpoint(wall, "end")
        if wall_start is None or wall_end is None:
            continue
        span = _opening_wall_span(opening_start, opening_end, wall_start, wall_end)
        if span is None or span <= best_span:
            continue
        best_span = span
        thickness = boundary_wall_thick if wall.get("is_exterior") else room_wall_thick
    return thickness


def _wall_thickness(
    wall: dict[str, Any],
    *,
    room_wall_thick: float,
    boundary_wall_thick: float,
) -> float:
    return boundary_wall_thick if wall.get("is_exterior") else room_wall_thick


def _draw_wall_strip(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    thickness: float,
) -> None:
    strip = _wall_strip_polygon(start, end, thickness)
    if strip:
        draw.polygon(strip, fill=1)


def _pillar_corners(
    cx: float,
    cy: float,
    width: float,
    height: float,
    rotation_deg: float,
) -> list[tuple[float, float]]:
    if width <= 0 or height <= 0:
        return []
    half_w, half_h = width / 2.0, height / 2.0
    rad = math.radians(rotation_deg)
    cos_t, sin_t = math.cos(rad), math.sin(rad)
    return [
        (
            cx + x * cos_t - y * sin_t,
            cy + x * sin_t + y * cos_t,
        )
        for x, y in ((-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h))
    ]


def _draw_pillar(draw: ImageDraw.ImageDraw, pillar: dict[str, Any]) -> None:
    try:
        cx = float(pillar["x"])
        cy = float(pillar["y"])
        width = float(pillar["width"])
        height = float(pillar["height"])
        rotation_deg = float(pillar.get("rotation_deg") or 0.0)
    except (KeyError, TypeError, ValueError):
        return
    corners = _pillar_corners(cx, cy, width, height, rotation_deg)
    if corners:
        draw.polygon(corners, fill=1)


def _wall_strip_polygon(
    start: tuple[float, float],
    end: tuple[float, float],
    thickness: float,
) -> list[tuple[float, float]]:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return []
    offset_x = -dy / length * (thickness / 2.0)
    offset_y = dx / length * (thickness / 2.0)
    return [
        (start[0] + offset_x, start[1] + offset_y),
        (end[0] + offset_x, end[1] + offset_y),
        (end[0] - offset_x, end[1] - offset_y),
        (start[0] - offset_x, start[1] - offset_y),
    ]


def _open_ring(polygon: Polygon) -> list[tuple[float, float]]:
    ring = [(float(x), float(y)) for x, y in polygon.exterior.coords]
    if len(ring) >= 2 and ring[0] == ring[-1]:
        return ring[:-1]
    return ring


def _edge_from_key(key: tuple[float, float, float, float]) -> tuple[tuple[float, float], tuple[float, float]]:
    return (key[0], key[1]), (key[2], key[3])


def _wall_endpoint(wall: dict[str, Any], key: str) -> tuple[float, float] | None:
    point = wall.get(key) or {}
    try:
        return (float(point["x"]), float(point["y"]))
    except (KeyError, TypeError, ValueError):
        return None


def _shared_wall_endpoints(walls: list[dict[str, Any]]) -> set[tuple[int, str]]:
    refs: list[tuple[int, str, tuple[float, float]]] = []
    for index, wall in enumerate(walls):
        for key in ("start", "end"):
            point = _wall_endpoint(wall, key)
            if point is not None:
                refs.append((index, key, point))

    tol2 = _GEOM_TOL * _GEOM_TOL
    shared: set[tuple[int, str]] = set()
    for a in range(len(refs)):
        index_a, key_a, (ax, ay) = refs[a]
        for b in range(a + 1, len(refs)):
            index_b, key_b, (bx, by) = refs[b]
            dx, dy = ax - bx, ay - by
            if dx * dx + dy * dy <= tol2:
                shared.add((index_a, key_a))
                shared.add((index_b, key_b))
    return shared


def _extend_wall_endpoints(
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    thickness: float,
    extend_start: bool,
    extend_end: bool,
) -> tuple[tuple[float, float], tuple[float, float]]:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return start, end
    half = thickness / 2.0
    ux, uy = dx / length, dy / length
    sx, sy = start
    ex, ey = end
    if extend_start:
        sx -= ux * half
        sy -= uy * half
    if extend_end:
        ex += ux * half
        ey += uy * half
    return (sx, sy), (ex, ey)


def _walls_uncut(extraction: dict[str, Any]) -> list[dict[str, Any]]:
    """Full wall segments from room edges, without door/window cuts."""
    rooms = list(extraction.get("rooms") or [])
    edge_counts: Counter[tuple[float, float, float, float]] = Counter()
    for room in rooms:
        polygon = _polygon_from_vertices(room["polygon"])
        if polygon.is_empty:
            continue
        ring = _open_ring(polygon)
        for index in range(len(ring)):
            edge_counts[_edge_key(ring[index], ring[(index + 1) % len(ring)])] += 1

    walls: list[dict[str, Any]] = []
    for key, count in edge_counts.items():
        start, end = _edge_from_key(key)
        walls.append(
            {
                "start": {"x": start[0], "y": start[1]},
                "end": {"x": end[0], "y": end[1]},
                "is_exterior": count < 2,
            }
        )
    return walls


def rasterize_wall_mask(
    walls: list[dict[str, Any]],
    size: int,
    *,
    room_wall_thick: float,
    boundary_wall_thick: float,
    pillars: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    """Thick wall strips from uncut edges, plus filled pillar boxes."""
    mask = Image.new("L", (size, size), 0)
    if not walls and not pillars:
        return np.zeros((size, size), dtype=np.float32)
    draw = ImageDraw.Draw(mask)
    shared = _shared_wall_endpoints(walls)
    for index, wall in enumerate(walls):
        start = _wall_endpoint(wall, "start")
        end = _wall_endpoint(wall, "end")
        if start is None or end is None:
            continue
        thickness = _wall_thickness(
            wall,
            room_wall_thick=room_wall_thick,
            boundary_wall_thick=boundary_wall_thick,
        )
        draw_start, draw_end = _extend_wall_endpoints(
            start,
            end,
            thickness=thickness,
            extend_start=(index, "start") in shared,
            extend_end=(index, "end") in shared,
        )
        _draw_wall_strip(draw, draw_start, draw_end, thickness)
    for pillar in pillars or []:
        _draw_pillar(draw, pillar)
    return np.asarray(mask, dtype=np.float32)


def rasterize_opening_mask(
    size: int,
    segments: list[dict[str, Any]],
    walls: list[dict[str, Any]],
    *,
    room_wall_thick: float,
    boundary_wall_thick: float,
) -> np.ndarray:
    """Door/window segments as wall strips on their host wall edge."""
    mask = Image.new("L", (size, size), 0)
    if not segments:
        return np.zeros((size, size), dtype=np.float32)
    draw = ImageDraw.Draw(mask)
    for segment in segments:
        start = (float(segment["start"]["x"]), float(segment["start"]["y"]))
        end = (float(segment["end"]["x"]), float(segment["end"]["y"]))
        thickness = _opening_wall_thickness(
            start,
            end,
            walls,
            room_wall_thick=room_wall_thick,
            boundary_wall_thick=boundary_wall_thick,
        )
        _draw_wall_strip(draw, start, end, thickness)
    return np.asarray(mask, dtype=np.float32)


def _wall_thicknesses(input_size: int) -> tuple[float, float]:
    scale = extraction_scale(input_size)
    return ROOM_WALL_THICK * scale, BOUNDARY_WALL_THICK * scale


def build_floorplan_targets(extraction: dict[str, Any], input_size: int) -> np.ndarray:
    """All walls, doors, and windows as three separate masks."""
    walls = _walls_uncut(extraction)
    doors, windows = _openings(extraction)
    room_wall_thick, boundary_wall_thick = _wall_thicknesses(input_size)
    wall_mask = rasterize_wall_mask(
        walls,
        input_size,
        room_wall_thick=room_wall_thick,
        boundary_wall_thick=boundary_wall_thick,
        pillars=list(extraction.get("pillars") or []),
    )
    door_mask = rasterize_opening_mask(
        input_size,
        doors,
        walls,
        room_wall_thick=room_wall_thick,
        boundary_wall_thick=boundary_wall_thick,
    )
    window_mask = rasterize_opening_mask(
        input_size,
        windows,
        walls,
        room_wall_thick=room_wall_thick,
        boundary_wall_thick=boundary_wall_thick,
    )
    return np.stack([wall_mask, door_mask, window_mask], axis=0)


class FloorplanDataset(IterableDataset):
    def __init__(
        self,
        dataset: Any,
        *,
        input_size: int,
        num_rows: int,
        seed: int | None = None,
    ) -> None:
        self._dataset = dataset
        self._input_size = input_size
        self._num_rows = num_rows
        self._seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return self._num_rows * 2

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        indices = list(range(self._num_rows))
        rng = random.Random((self._seed or 0) + self.epoch)
        rng.shuffle(indices)
        scale = extraction_scale(self._input_size)
        for dataset_index in indices:
            row = self._dataset[dataset_index]
            extraction = scale_extraction(json.loads(str(row["extraction"])), scale)
            targets = torch.from_numpy(build_floorplan_targets(extraction, self._input_size))
            for key in ("image", "simple_image"):
                image = prepare_image(row[key], self._input_size)
                yield image_to_tensor(image), targets


class FloorplanCollator:
    def __call__(self, batch: list[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, torch.Tensor]:
        images, masks = zip(*batch, strict=True)
        return {
            "images": torch.stack(list(images)),
            "masks": torch.stack(list(masks)),
        }
