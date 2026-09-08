"""Hub rows, room point sampling, mask rasterization, and SAM collator."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import torch
from PIL import Image, ImageDraw
from registry import hub_dataset_row_count, load_hub_rows_at, load_hub_stream
from shapely.geometry import Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from torch.utils.data import IterableDataset

SAM_COLUMNS = ("stem", "image", "extraction")
STREAM_SHUFFLE_BUFFER = 256
ROOM_WALL_THICK = 4.0
BOUNDARY_WALL_THICK = 8.0


@dataclass(frozen=True)
class VizSample:
    stem: str
    image: Image.Image
    point: tuple[float, float]
    pixel_values: torch.Tensor


def prepare_image(image: Image.Image, size: int) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.LANCZOS)
    return image


def load_sample_rows(
    dataset: str,
    *,
    count: int,
    seed: int | None = None,
    revision: str | None = None,
    cache_dir: Any = None,
) -> tuple[list[dict[str, Any]], list[int], int]:
    total = hub_dataset_row_count(dataset, revision=revision, cache_dir=cache_dir)
    if total == 0:
        raise ValueError(f"dataset {dataset!r} has no rows")
    n = min(count, total)
    rng = random.Random(seed)
    indices = rng.sample(range(total), n)
    rows = load_hub_rows_at(
        dataset,
        indices,
        columns=SAM_COLUMNS,
        revision=revision,
        cache_dir=cache_dir,
    )
    return rows, indices, total


def load_train_stream(
    dataset: str,
    *,
    test_fraction: float,
    revision: str | None = None,
    cache_dir: Any = None,
) -> tuple[Any, int, int]:
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

    train_rows = load_hub_stream(
        dataset,
        split="train",
        revision=revision,
        cache_dir=cache_dir,
        columns=SAM_COLUMNS,
    ).take(train_count)
    return train_rows, train_count, test_count


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
        columns=SAM_COLUMNS,
        revision=revision,
        cache_dir=cache_dir,
    )


def _polygon_from_vertices(vertices: list[dict[str, Any]]) -> Polygon:
    ring = [(float(point["x"]), float(point["y"])) for point in vertices]
    return Polygon(ring)


def _edge_key(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float, float, float]:
    if a > b:
        a, b = b, a
    return (round(a[0], 1), round(a[1], 1), round(b[0], 1), round(b[1], 1))


def _wall_thickness_by_edge(walls: list[dict[str, Any]]) -> dict[tuple[float, float, float, float], float]:
    lookup: dict[tuple[float, float, float, float], float] = {}
    for wall in walls:
        start = (float(wall["start"]["x"]), float(wall["start"]["y"]))
        end = (float(wall["end"]["x"]), float(wall["end"]["y"]))
        thick = BOUNDARY_WALL_THICK if wall.get("is_exterior") else ROOM_WALL_THICK
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


def _rooms_union(rooms: list[dict[str, Any]]) -> BaseGeometry:
    polygons = [_polygon_from_vertices(room["polygon"]) for room in rooms]
    return unary_union(polygons)


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


def sample_point_outside(geometry: BaseGeometry, rng: random.Random, canvas_size: int) -> tuple[float, float]:
    limit = float(canvas_size)
    while True:
        x = rng.uniform(0.0, limit)
        y = rng.uniform(0.0, limit)
        if not geometry.contains(Point(x, y)):
            return (x, y)


def rasterize_room_mask(room: dict[str, Any], walls: list[dict[str, Any]], size: int) -> np.ndarray:
    polygon = _polygon_from_vertices(room["polygon"])
    mask = Image.new("L", (size, size), 0)
    if polygon.is_empty:
        return np.zeros((size, size), dtype=np.float32)

    draw = ImageDraw.Draw(mask)
    exterior = [(float(x), float(y)) for x, y in polygon.exterior.coords]
    draw.polygon(exterior, fill=1)

    thickness_by_edge = _wall_thickness_by_edge(walls)
    ring = _open_ring(polygon)
    for index in range(len(ring)):
        start = ring[index]
        end = ring[(index + 1) % len(ring)]
        thickness = thickness_by_edge.get(_edge_key(start, end), ROOM_WALL_THICK)
        strip = _wall_strip_polygon(start, end, thickness)
        if strip:
            draw.polygon(strip, fill=0)

    return np.asarray(mask, dtype=np.float32)


def build_room_samples(
    extraction: dict[str, Any],
    index: int,
    *,
    canvas_size: int,
    room_slots: int,
    per_room_variation: int,
    empty_mask_prob: float,
    point_distance_alpha: float,
    seed: int | None,
) -> list[tuple[tuple[float, float], np.ndarray]]:
    rooms = list(extraction.get("rooms") or [])
    if not rooms:
        raise ValueError("extraction has no rooms")

    walls = list(extraction.get("walls") or [])
    rooms_union = _rooms_union(rooms)
    rng = random.Random((seed or 0) * 1_000_003 + index)
    samples: list[tuple[tuple[float, float], np.ndarray]] = []
    for _ in range(room_slots):
        room = rng.choice(rooms)
        room_mask = rasterize_room_mask(room, walls, canvas_size)
        distance = _mask_distance_transform(room_mask)
        for _ in range(per_room_variation):
            if rng.random() < empty_mask_prob:
                point = sample_point_outside(rooms_union, rng, canvas_size)
                mask = np.zeros((canvas_size, canvas_size), dtype=np.float32)
            else:
                point = sample_point_from_mask(
                    room_mask,
                    rng,
                    alpha=point_distance_alpha,
                    distance=distance,
                )
                mask = room_mask
            samples.append((point, mask))
    return samples


def build_viz_samples(
    rows: list[dict[str, Any]],
    canvas_size: int,
    *,
    room_slots: int,
    per_room_variation: int,
    empty_mask_prob: float,
    point_distance_alpha: float,
    seed: int | None,
    prepare_pixel_values: Any,
) -> list[VizSample]:
    samples: list[VizSample] = []
    for index, row in enumerate(rows):
        image = prepare_image(row["image"], canvas_size)
        extraction = json.loads(str(row["extraction"]))
        point, _mask = build_room_samples(
            extraction,
            index,
            canvas_size=canvas_size,
            room_slots=room_slots,
            per_room_variation=per_room_variation,
            empty_mask_prob=empty_mask_prob,
            point_distance_alpha=point_distance_alpha,
            seed=seed,
        )[0]
        samples.append(
            VizSample(
                stem=str(row["stem"]),
                image=image,
                point=point,
                pixel_values=prepare_pixel_values(image),
            )
        )
    return samples


class FloorplanSamDataset(IterableDataset):
    def __init__(
        self,
        rows: Any,
        cfg_canvas_size: int,
        *,
        num_rows: int,
        room_slots: int,
        per_room_variation: int,
        empty_mask_prob: float,
        point_distance_alpha: float,
        seed: int | None,
        prepare_pixel_values: Any,
    ) -> None:
        self._rows = rows
        self._canvas_size = cfg_canvas_size
        self._num_rows = num_rows
        self._room_slots = room_slots
        self._per_room_variation = per_room_variation
        self._empty_mask_prob = empty_mask_prob
        self._point_distance_alpha = point_distance_alpha
        self._seed = seed or 0
        self._prepare_pixel_values = prepare_pixel_values
        self.epoch = 0

    def __len__(self) -> int:
        return self._num_rows * self._room_slots * self._per_room_variation

    def __iter__(self) -> Iterator[dict[str, Any]]:
        buffer_size = min(STREAM_SHUFFLE_BUFFER, self._num_rows)
        shuffled = self._rows.shuffle(seed=self._seed + self.epoch, buffer_size=buffer_size)
        for index, row in enumerate(shuffled):
            if index >= self._num_rows:
                break
            image = prepare_image(row["image"], self._canvas_size)
            extraction = json.loads(str(row["extraction"]))
            samples = build_room_samples(
                extraction,
                index,
                canvas_size=self._canvas_size,
                room_slots=self._room_slots,
                per_room_variation=self._per_room_variation,
                empty_mask_prob=self._empty_mask_prob,
                point_distance_alpha=self._point_distance_alpha,
                seed=self._seed,
            )

            shared = {"pixel_values": self._prepare_pixel_values(image)}

            for point, mask in samples:
                yield {
                    **shared,
                    "point": point,
                    "mask": mask,
                }


class SamCollator:
    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        points = torch.tensor(
            [[[[x, y] for x, y in (item["point"],)]] for item in batch],
            dtype=torch.float32,
        )
        labels_pts = torch.ones((len(batch), 1, 1), dtype=torch.int64)
        masks = torch.from_numpy(np.stack([item["mask"] for item in batch])).float()

        return {
            "input_points": points,
            "input_labels": labels_pts,
            "labels": masks,
            "pixel_values": torch.cat([item["pixel_values"] for item in batch], dim=0),
        }
