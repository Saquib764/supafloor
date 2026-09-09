"""Hub stream → starred RGB input + 3-channel room structure target."""

from __future__ import annotations

import json
import math
import random
from copy import deepcopy
from typing import Any, Iterator

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from registry import hub_dataset_row_count, load_hub_dataset, load_hub_rows_at
from shapely.geometry import Polygon
from torch.utils.data import IterableDataset

from config import Config
from star import draw_filled_star, star_inside_mask

HUB_COLUMNS = ("stem", "image", "extraction")
_EXTRACTION_CANVAS = 1024
ROOM_WALL_THICK = 4.0


def extraction_scale(input_size: int) -> float:
    return input_size / _EXTRACTION_CANVAS


BOUNDARY_WALL_THICK = 8.0
_STAR_TRIES = 10


def prepare_image(image: Image.Image, size: int) -> Image.Image:
    """Grayscale → invert → RGB (white walls on black background)."""
    gray = image.convert("L")
    if gray.size != (size, size):
        gray = gray.resize((size, size), Image.Resampling.LANCZOS)
    return ImageOps.invert(gray).convert("RGB")


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


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


def _doors_for_room(room_name: str, doors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [door for door in doors if room_name in (door.get("connects") or [])]


def _windows_for_room(room_name: str, windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [window for window in windows if room_name in (window.get("room_names") or [])]


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


def _mask_distance_transform(mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt

    return distance_transform_edt(mask > 0)


def sample_point_from_mask(
    mask: np.ndarray,
    rng: random.Random,
    *,
    alpha: float,
    distance: np.ndarray | None = None,
) -> tuple[float, float]:
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        raise ValueError("cannot sample point from empty mask")
    dist = distance if distance is not None else _mask_distance_transform(mask)
    weights = dist[ys, xs] ** alpha
    total = float(weights.sum())
    if total <= 0.0:
        index = rng.randrange(len(xs))
        return (float(xs[index]), float(ys[index]))
    index = rng.choices(range(len(xs)), weights=(weights / total).tolist(), k=1)[0]
    return (float(xs[index]), float(ys[index]))


def rasterize_room_mask(
    room: dict[str, Any],
    walls: list[dict[str, Any]],
    size: int,
    *,
    room_wall_thick: float,
    boundary_wall_thick: float,
) -> np.ndarray:
    polygon = _polygon_from_vertices(room["polygon"])
    mask = Image.new("L", (size, size), 0)
    if polygon.is_empty:
        return np.zeros((size, size), dtype=np.float32)
    draw = ImageDraw.Draw(mask)
    exterior = [(float(x), float(y)) for x, y in polygon.exterior.coords]
    draw.polygon(exterior, fill=1)
    thickness_by_edge = _wall_thickness_by_edge(
        walls,
        room_wall_thick=room_wall_thick,
        boundary_wall_thick=boundary_wall_thick,
    )
    ring = _open_ring(polygon)
    for index in range(len(ring)):
        start = ring[index]
        end = ring[(index + 1) % len(ring)]
        thickness = thickness_by_edge.get(_edge_key(start, end), room_wall_thick)
        strip = _wall_strip_polygon(start, end, thickness)
        if strip:
            draw.polygon(strip, fill=0)
    return np.asarray(mask, dtype=np.float32)


def room_boundary_mask(room_mask: np.ndarray, *, low: int = 50, high: int = 150) -> np.ndarray:
    """Canny edges of the filled room interior mask."""
    if not np.any(room_mask > 0):
        return np.zeros_like(room_mask, dtype=np.float32)
    gray = (np.clip(room_mask, 0.0, 1.0) * 255.0).astype(np.uint8)
    edges = cv2.Canny(gray, low, high)
    return (edges > 0).astype(np.float32)


def rasterize_line_mask(
    size: int,
    segments: list[dict[str, Any]],
    *,
    line_width: float,
) -> np.ndarray:
    mask = Image.new("L", (size, size), 0)
    if not segments:
        return np.zeros((size, size), dtype=np.float32)
    draw = ImageDraw.Draw(mask)
    width = max(1, int(round(line_width)))
    for segment in segments:
        start = segment["start"]
        end = segment["end"]
        draw.line(
            [
                (float(start["x"]), float(start["y"])),
                (float(end["x"]), float(end["y"])),
            ],
            fill=1,
            width=width,
        )
    return np.asarray(mask, dtype=np.float32)


def build_room_targets(
    room: dict[str, Any],
    extraction: dict[str, Any],
    size: int,
    cfg: Config,
) -> np.ndarray:
    walls = list(extraction.get("walls") or [])
    doors, windows = _openings(extraction)
    room_name = str(room.get("name") or "")
    scale = extraction_scale(cfg.input_size)
    room_wall_thick = ROOM_WALL_THICK * scale
    boundary_wall_thick = BOUNDARY_WALL_THICK * scale
    interior = rasterize_room_mask(
        room,
        walls,
        size,
        room_wall_thick=room_wall_thick,
        boundary_wall_thick=boundary_wall_thick,
    )
    boundary = room_boundary_mask(interior)
    door_mask = rasterize_line_mask(
        size,
        _doors_for_room(room_name, doors),
        line_width=cfg.data.door_line_width * scale,
    )
    window_mask = rasterize_line_mask(
        size,
        _windows_for_room(room_name, windows),
        line_width=cfg.data.window_line_width * scale,
    )
    return np.stack([boundary, door_mask, window_mask], axis=0)


def _sample_star(
    mask: np.ndarray,
    rng: random.Random,
    *,
    min_radius: float,
    max_radius: float,
    inner_ratio: float,
    point_alpha: float,
) -> tuple[float, float, float]:
    distance = _mask_distance_transform(mask)
    cx, cy = sample_point_from_mask(mask, rng, alpha=point_alpha, distance=distance)
    radius = rng.uniform(min_radius, max_radius)
    for _ in range(_STAR_TRIES):
        if star_inside_mask(mask, cx, cy, radius, inner_ratio=inner_ratio):
            return cx, cy, radius
        radius = max(min_radius, radius * 0.85)
        cx, cy = sample_point_from_mask(mask, rng, alpha=point_alpha, distance=distance)
    return cx, cy, max(min_radius, radius * 0.5)


def build_star_samples(
    extraction: dict[str, Any],
    index: int,
    cfg: Config,
) -> list[tuple[tuple[float, float, float], np.ndarray]]:
    rooms = list(extraction.get("rooms") or [])
    if not rooms:
        raise ValueError("extraction has no rooms")
    walls = list(extraction.get("walls") or [])
    scale = extraction_scale(cfg.input_size)
    room_wall_thick = ROOM_WALL_THICK * scale
    boundary_wall_thick = BOUNDARY_WALL_THICK * scale
    rng = random.Random((cfg.data.seed or 0) * 1_000_003 + index)
    samples: list[tuple[tuple[float, float, float], np.ndarray]] = []
    for _ in range(cfg.data.room_slots):
        room = rng.choice(rooms)
        room_mask = rasterize_room_mask(
            room,
            walls,
            cfg.input_size,
            room_wall_thick=room_wall_thick,
            boundary_wall_thick=boundary_wall_thick,
        )
        for _ in range(cfg.data.per_room_variation):
            star = _sample_star(
                room_mask,
                rng,
                min_radius=cfg.star.min_radius,
                max_radius=cfg.star.max_radius,
                inner_ratio=cfg.star.inner_ratio,
                point_alpha=cfg.data.point_distance_alpha,
            )
            targets = build_room_targets(room, extraction, cfg.input_size, cfg)
            samples.append((star, targets))
    return samples


class FloorplanStarDataset(IterableDataset):
    def __init__(
        self,
        dataset: Any,
        cfg: Config,
        *,
        num_rows: int,
    ) -> None:
        self._dataset = dataset
        self._cfg = cfg
        self._num_rows = num_rows
        self.epoch = 0

    def __len__(self) -> int:
        return self._num_rows * self._cfg.data.variations_per_floorplan

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        indices = list(range(self._num_rows))
        rng = random.Random((self._cfg.data.seed or 0) + self.epoch)
        rng.shuffle(indices)
        for row_index, dataset_index in enumerate(indices):
            row = self._dataset[dataset_index]
            base = prepare_image(row["image"], self._cfg.input_size)
            scale = extraction_scale(self._cfg.input_size)
            extraction = scale_extraction(json.loads(str(row["extraction"])), scale)
            for (cx, cy, radius), mask in build_star_samples(extraction, row_index, self._cfg):
                starred = base.copy()
                draw_filled_star(
                    starred,
                    cx,
                    cy,
                    radius,
                    color=self._cfg.star.color,
                    inner_ratio=self._cfg.star.inner_ratio,
                )
                yield image_to_tensor(starred), torch.from_numpy(mask)


class StarCollator:
    def __call__(self, batch: list[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, torch.Tensor]:
        images, masks = zip(*batch, strict=True)
        return {
            "images": torch.stack(list(images)),
            "masks": torch.stack(list(masks)),
        }
