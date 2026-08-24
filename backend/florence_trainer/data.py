"""Hub rows, training/infer sample builders, and Florence collator."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset as HubDataset
from PIL import Image
from shapely.geometry import Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from loc import format_loc_point
from registry import load_hub_dataset, load_hub_samples
from torch.utils.data import Dataset

_VARIATIONS_PER_ROOM = 2


def prepare_image(image: Image.Image, size: int) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.LANCZOS)
    return image


def load_rows(dataset: str, *, revision: str | None = None, cache_dir: Path | None = None) -> HubDataset:
    return load_hub_dataset(dataset, revision=revision, cache_dir=cache_dir)


def load_sample_rows(
    dataset: str,
    *,
    count: int,
    seed: int | None = None,
    revision: str | None = None,
    cache_dir: Path | None = None,
) -> tuple[list[dict[str, object]], list[int], int]:
    return load_hub_samples(
        dataset,
        count=count,
        seed=seed,
        revision=revision,
        cache_dir=cache_dir,
    )


def _polygon_from_vertices(vertices: list[dict[str, Any]]) -> Polygon:
    ring = [(float(point["x"]), float(point["y"])) for point in vertices]
    return Polygon(ring)


def _room_polygons(rooms: list[dict[str, Any]]) -> list[Polygon]:
    return [_polygon_from_vertices(room["polygon"]) for room in rooms]


def _rooms_union(rooms: list[dict[str, Any]]) -> BaseGeometry:
    return unary_union(_room_polygons(rooms))


def polygon_to_canvas_ring(
    polygon: list[list[int | float]],
    *,
    image_size: int,
    canvas_size: int,
) -> list[tuple[float, float]]:
    scale = canvas_size / image_size
    return [(float(x) * scale, float(y) * scale) for x, y in polygon]


def sample_points(
    polygon: BaseGeometry,
    rng: random.Random,
    *,
    canvas_size: int,
    inside_count: int,
    outside_count: int,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    limit = float(canvas_size)
    inside_points: list[tuple[float, float]] = []
    outside_points: list[tuple[float, float]] = []

    if inside_count:
        if polygon.is_empty:
            raise ValueError("cannot sample inside points from empty polygon")
        minx, miny, maxx, maxy = polygon.bounds
        while len(inside_points) < inside_count:
            x = rng.uniform(minx, maxx)
            y = rng.uniform(miny, maxy)
            if polygon.contains(Point(x, y)):
                inside_points.append((x, y))

    if outside_count:
        while len(outside_points) < outside_count:
            x = rng.uniform(0.0, limit)
            y = rng.uniform(0.0, limit)
            if not polygon.contains(Point(x, y)):
                outside_points.append((x, y))

    return inside_points, outside_points


def format_label(
    room: dict[str, Any] | None,
    *,
    canvas_size: int,
    bins: int,
) -> str:
    if room is None:
        return "<poly></poly>"
    vertex_locs = "".join(
        format_loc_point(float(point["x"]), float(point["y"]), canvas_size=canvas_size, bins=bins)
        for point in room["polygon"]
    )
    return f"<poly>{vertex_locs}</poly>"


def format_prompt(
    positive_points: list[tuple[float, float]],
    *,
    canvas_size: int,
    bins: int,
    excluded_points: list[tuple[float, float]] | None = None,
) -> str:
    positive_locs = "".join(
        format_loc_point(x, y, canvas_size=canvas_size, bins=bins)
        for x, y in positive_points
    )
    prompt = f"Locate room polygon containing point {positive_locs}"
    if excluded_points:
        excluded_locs = "".join(
            format_loc_point(x, y, canvas_size=canvas_size, bins=bins)
            for x, y in excluded_points
        )
        prompt += f", and not containing {excluded_locs}"
    return prompt


@dataclass(frozen=True)
class ReferringDatum:
    prompt: str
    label: str
    positive_points: list[tuple[float, float]]
    excluded_points: list[tuple[float, float]]


def build_referring_variations(
    extraction: dict[str, Any],
    index: int,
    *,
    canvas_size: int,
    bins: int,
    seed: int | None,
    empty_polygon_prob: float,
    excluded_point_prob: float,
    variations_per_floorplan: int,
) -> list[ReferringDatum]:
    if variations_per_floorplan % _VARIATIONS_PER_ROOM != 0:
        raise ValueError(
            f"variations_per_floorplan ({variations_per_floorplan}) must be divisible by "
            f"{_VARIATIONS_PER_ROOM} (variations per room)"
        )

    rooms = list(extraction.get("rooms") or [])
    if not rooms:
        raise ValueError("extraction has no rooms")

    room_slots = variations_per_floorplan // _VARIATIONS_PER_ROOM
    rooms_union = _rooms_union(rooms)
    rng = random.Random((seed or 0) * 1_000_003 + index)
    selected_rooms = rng.choices(rooms, k=room_slots)

    variations: list[ReferringDatum] = []
    for room in selected_rooms:
        for _ in range(_VARIATIONS_PER_ROOM):
            excluded_count = rng.randint(1, 6) if rng.random() < excluded_point_prob else 0
            empty_polygon = rng.random() < empty_polygon_prob

            if empty_polygon:
                polygon = rooms_union
                inside_count = 0
                outside_count = 1
                label_room = None
            else:
                polygon = _polygon_from_vertices(room["polygon"])
                inside_count = rng.randint(1, 6)
                outside_count = excluded_count
                label_room = room

            inside, outside = sample_points(
                polygon,
                rng,
                canvas_size=canvas_size,
                inside_count=inside_count,
                outside_count=outside_count,
            )
            if empty_polygon:
                positive_points = outside
                excluded_points: list[tuple[float, float]] = []
            else:
                positive_points = inside
                excluded_points = outside

            prompt = format_prompt(
                positive_points,
                canvas_size=canvas_size,
                bins=bins,
                excluded_points=excluded_points,
            )
            label = format_label(
                label_room,
                canvas_size=canvas_size,
                bins=bins,
            )
            variations.append(
                ReferringDatum(
                    prompt=prompt,
                    label=label,
                    positive_points=positive_points,
                    excluded_points=excluded_points,
                )
            )

    return variations


def room_overlaps_existing(
    room: dict[str, Any],
    canvas_rings: list[list[tuple[float, float]]],
    *,
    image_size: int,
    canvas_size: int,
) -> bool:
    polygon = room.get("polygon") or []
    if len(polygon) < 3 or not canvas_rings:
        return False
    cx = sum(point[0] for point in polygon) / len(polygon)
    cy = sum(point[1] for point in polygon) / len(polygon)
    scale = canvas_size / image_size
    point = Point(cx * scale, cy * scale)
    return any(
        existing_poly.contains(point)
        for existing_poly in (Polygon(ring) for ring in canvas_rings if len(ring) >= 3)
    )


def format_room_plot_block(
    name: str,
    polygon: list[list[int | float]],
    *,
    image_size: int,
    canvas_size: int,
    bins: int,
) -> str:
    scale = canvas_size / image_size
    vertex_locs = "".join(
        format_loc_point(float(x) * scale, float(y) * scale, canvas_size=canvas_size, bins=bins)
        for x, y in polygon
    )
    return f"{name.strip() or 'Room'}\n<poly>{vertex_locs}</poly>"


def format_plot_text(
    rooms: list[dict[str, Any]],
    *,
    image_size: int,
    canvas_size: int,
    bins: int,
) -> str:
    blocks = [
        format_room_plot_block(
            str(room.get("name") or "Room"),
            room.get("polygon") or [],
            image_size=image_size,
            canvas_size=canvas_size,
            bins=bins,
        )
        for room in rooms
        if len(room.get("polygon") or []) >= 3
    ]
    return "\n\n".join(blocks) + ("\n" if blocks else "")


class FloorplanFlorenceDataset(Dataset):
    def __init__(
        self,
        rows: HubDataset,
        image_size: int,
        *,
        loc_bins: int,
        canvas_size: int,
        empty_polygon_prob: float,
        excluded_point_prob: float,
        variations_per_floorplan: int,
        seed: int | None = None,
    ) -> None:
        self._rows = rows
        self._image_size = image_size
        self._loc_bins = loc_bins
        self._canvas_size = canvas_size
        self._empty_polygon_prob = empty_polygon_prob
        self._excluded_point_prob = excluded_point_prob
        self._variations_per_floorplan = variations_per_floorplan
        self._seed = seed

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> list[dict[str, object]]:
        row = self._rows[index]
        extraction = json.loads(str(row["extraction"]))
        image = prepare_image(row["image"], self._image_size)
        variations = build_referring_variations(
            extraction,
            index,
            canvas_size=self._canvas_size,
            bins=self._loc_bins,
            seed=self._seed,
            empty_polygon_prob=self._empty_polygon_prob,
            excluded_point_prob=self._excluded_point_prob,
            variations_per_floorplan=self._variations_per_floorplan,
        )
        return [
            {
                "image": image,
                "prompt": datum.prompt,
                "label": datum.label,
            }
            for datum in variations
        ]


class FlorenceCollator:
    def __init__(self, processor: object, max_label_tokens: int) -> None:
        self._processor = processor
        self._max_label_tokens = max_label_tokens

    def __call__(self, batch: list[dict[str, object] | list[dict[str, object]]]) -> dict[str, object]:
        flat: list[dict[str, object]] = []
        for item in batch:
            if isinstance(item, list):
                flat.extend(item)
            else:
                flat.append(item)
        inputs = self._processor(
            text=[item["prompt"] for item in flat],
            images=[item["image"] for item in flat],
            return_tensors="pt",
            padding=True,
        )
        labels = self._processor.tokenizer(
            [item["label"] for item in flat],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_label_tokens,
        ).input_ids
        labels[labels == self._processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        return inputs
