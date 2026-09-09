"""Hub rows → grayscale floor plan input + 3-channel structure targets."""

from __future__ import annotations

import json
import math
import random
from copy import deepcopy
from typing import Any, Iterator

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from registry import hub_dataset_row_count, load_hub_dataset, load_hub_rows_at
from shapely.geometry import Polygon
from torch.utils.data import IterableDataset

HUB_COLUMNS = ("stem", "image", "extraction")
_EXTRACTION_CANVAS = 1024
ROOM_WALL_THICK = 4.0
BOUNDARY_WALL_THICK = 8.0


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
    for wall in scaled.get("walls") or []:
        wall["start"] = _scale_point(wall["start"], scale)
        wall["end"] = _scale_point(wall["end"], scale)
    openings = scaled.get("openings") or {}
    scaled["openings"] = {
        "doors": [_scale_segment(door, scale) for door in openings.get("doors") or []],
        "windows": [_scale_segment(window, scale) for window in openings.get("windows") or []],
    }
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


def load_viz_rows(
    dataset: str,
    train_count: int,
    *,
    count: int,
    revision: str | None = None,
    cache_dir: Any = None,
) -> list[dict[str, Any]]:
    if count < 1:
        return []
    indices = range(train_count, train_count + count)
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


def _wall_thickness_by_edge(
    walls: list[dict[str, Any]],
    *,
    room_wall_thick: float,
    boundary_wall_thick: float,
) -> dict[tuple[float, float, float, float], float]:
    lookup: dict[tuple[float, float, float, float], float] = {}
    for wall in walls:
        start = (float(wall["start"]["x"]), float(wall["start"]["y"]))
        end = (float(wall["end"]["x"]), float(wall["end"]["y"]))
        thick = boundary_wall_thick if wall.get("is_exterior") else room_wall_thick
        lookup[_edge_key(start, end)] = thick
    return lookup


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


def _draw_wall_strips(
    draw: ImageDraw.ImageDraw,
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
    thickness_by_edge: dict[tuple[float, float, float, float], float],
    *,
    default_thickness: float,
) -> None:
    for start, end in segments:
        thickness = thickness_by_edge.get(_edge_key(start, end), default_thickness)
        strip = _wall_strip_polygon(start, end, thickness)
        if strip:
            draw.polygon(strip, fill=1)


def rasterize_room_wall_mask(
    room: dict[str, Any],
    walls: list[dict[str, Any]],
    size: int,
    *,
    room_wall_thick: float,
    boundary_wall_thick: float,
) -> np.ndarray:
    """Wall strips along one room polygon edge."""
    polygon = _polygon_from_vertices(room["polygon"])
    mask = Image.new("L", (size, size), 0)
    if polygon.is_empty:
        return np.zeros((size, size), dtype=np.float32)
    draw = ImageDraw.Draw(mask)
    thickness_by_edge = _wall_thickness_by_edge(
        walls,
        room_wall_thick=room_wall_thick,
        boundary_wall_thick=boundary_wall_thick,
    )
    ring = _open_ring(polygon)
    segments = [
        (ring[index], ring[(index + 1) % len(ring)])
        for index in range(len(ring))
    ]
    _draw_wall_strips(draw, segments, thickness_by_edge, default_thickness=room_wall_thick)
    return np.asarray(mask, dtype=np.float32)


def rasterize_segment_wall_mask(
    size: int,
    segments: list[dict[str, Any]],
    walls: list[dict[str, Any]],
    *,
    room_wall_thick: float,
    boundary_wall_thick: float,
) -> np.ndarray:
    """Line segments as wall strips with matching thickness."""
    mask = Image.new("L", (size, size), 0)
    if not segments:
        return np.zeros((size, size), dtype=np.float32)
    draw = ImageDraw.Draw(mask)
    thickness_by_edge = _wall_thickness_by_edge(
        walls,
        room_wall_thick=room_wall_thick,
        boundary_wall_thick=boundary_wall_thick,
    )
    edge_segments = [
        (
            (float(segment["start"]["x"]), float(segment["start"]["y"])),
            (float(segment["end"]["x"]), float(segment["end"]["y"])),
        )
        for segment in segments
    ]
    _draw_wall_strips(draw, edge_segments, thickness_by_edge, default_thickness=room_wall_thick)
    return np.asarray(mask, dtype=np.float32)


def _wall_thicknesses(input_size: int) -> tuple[float, float]:
    scale = extraction_scale(input_size)
    return ROOM_WALL_THICK * scale, BOUNDARY_WALL_THICK * scale


def build_floorplan_targets(extraction: dict[str, Any], input_size: int) -> np.ndarray:
    """All walls, doors, and windows as three separate masks."""
    walls = list(extraction.get("walls") or [])
    doors, windows = _openings(extraction)
    room_wall_thick, boundary_wall_thick = _wall_thicknesses(input_size)
    if walls:
        wall_mask = rasterize_segment_wall_mask(
            input_size,
            walls,
            walls,
            room_wall_thick=room_wall_thick,
            boundary_wall_thick=boundary_wall_thick,
        )
    else:
        wall_mask = np.zeros((input_size, input_size), dtype=np.float32)
        for room in extraction.get("rooms") or []:
            wall_mask = np.maximum(
                wall_mask,
                rasterize_room_wall_mask(
                    room,
                    walls,
                    input_size,
                    room_wall_thick=room_wall_thick,
                    boundary_wall_thick=boundary_wall_thick,
                ),
            )
    door_mask = rasterize_segment_wall_mask(
        input_size,
        doors,
        walls,
        room_wall_thick=room_wall_thick,
        boundary_wall_thick=boundary_wall_thick,
    )
    window_mask = rasterize_segment_wall_mask(
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
        return self._num_rows

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        indices = list(range(self._num_rows))
        rng = random.Random((self._seed or 0) + self.epoch)
        rng.shuffle(indices)
        scale = extraction_scale(self._input_size)
        for dataset_index in indices:
            row = self._dataset[dataset_index]
            image = prepare_image(row["image"], self._input_size)
            extraction = scale_extraction(json.loads(str(row["extraction"])), scale)
            targets = build_floorplan_targets(extraction, self._input_size)
            yield image_to_tensor(image), torch.from_numpy(targets)


class FloorplanCollator:
    def __call__(self, batch: list[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, torch.Tensor]:
        images, masks = zip(*batch, strict=True)
        return {
            "images": torch.stack(list(images)),
            "masks": torch.stack(list(masks)),
        }
