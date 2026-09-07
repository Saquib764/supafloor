"""Hub rows, room point sampling, mask rasterization, and SAM collator."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import torch
from PIL import Image, ImageDraw
from registry import hub_dataset_row_count, load_hub_rows_at, load_hub_stream
from shapely.geometry import Point, Polygon
from torch.utils.data import IterableDataset

SAM_COLUMNS = ("stem", "image", "extraction")
STREAM_SHUFFLE_BUFFER = 10_000


@dataclass(frozen=True)
class VizSample:
    stem: str
    image: Image.Image
    point: tuple[float, float]
    intermediate_hidden_states: tuple[torch.Tensor, ...] | None = None
    pixel_values: torch.Tensor | None = None


def prepare_image(image: Image.Image, size: int) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.LANCZOS)
    return image


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


def sample_point(polygon: Polygon, rng: random.Random) -> tuple[float, float]:
    if polygon.is_empty:
        raise ValueError("cannot sample point from empty polygon")
    minx, miny, maxx, maxy = polygon.bounds
    while True:
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        if polygon.contains(Point(x, y)):
            return (x, y)


def rasterize_mask(polygon: Polygon, size: int) -> np.ndarray:
    mask = Image.new("L", (size, size), 0)
    if polygon.is_empty:
        return np.zeros((size, size), dtype=np.float32)
    exterior = [(float(x), float(y)) for x, y in polygon.exterior.coords]
    ImageDraw.Draw(mask).polygon(exterior, outline=1, fill=1)
    return np.asarray(mask, dtype=np.float32)


def build_room_samples(
    extraction: dict[str, Any],
    index: int,
    *,
    canvas_size: int,
    room_slots: int,
    per_room_variation: int,
    seed: int | None,
) -> list[tuple[tuple[float, float], np.ndarray]]:
    rooms = list(extraction.get("rooms") or [])
    if not rooms:
        raise ValueError("extraction has no rooms")

    rng = random.Random((seed or 0) * 1_000_003 + index)
    samples: list[tuple[tuple[float, float], np.ndarray]] = []
    for _ in range(room_slots):
        room = rng.choice(rooms)
        polygon = _polygon_from_vertices(room["polygon"])
        mask = rasterize_mask(polygon, canvas_size)
        for _ in range(per_room_variation):
            point = sample_point(polygon, rng)
            samples.append((point, mask))
    return samples


def build_viz_samples(
    rows: list[dict[str, Any]],
    canvas_size: int,
    *,
    room_slots: int,
    per_room_variation: int,
    seed: int | None,
    precompute_backbone: bool,
    encode_backbone: Any,
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
            seed=seed,
        )[0]
        shared: dict[str, Any] = {}
        if precompute_backbone:
            pixel_values = prepare_pixel_values(image)
            shared["intermediate_hidden_states"] = encode_backbone(pixel_values)
        else:
            shared["pixel_values"] = prepare_pixel_values(image)
        samples.append(
            VizSample(
                stem=str(row["stem"]),
                image=image,
                point=point,
                **shared,
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
        seed: int | None,
        precompute_backbone: bool,
        encode_backbone: Any,
        prepare_pixel_values: Any,
    ) -> None:
        self._rows = rows
        self._canvas_size = cfg_canvas_size
        self._num_rows = num_rows
        self._room_slots = room_slots
        self._per_room_variation = per_room_variation
        self._seed = seed or 0
        self._precompute_backbone = precompute_backbone
        self._encode_backbone = encode_backbone
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
                seed=self._seed,
            )

            shared: dict[str, Any] = {}
            if self._precompute_backbone:
                pixel_values = self._prepare_pixel_values(image)
                shared["intermediate_hidden_states"] = self._encode_backbone(pixel_values)
            else:
                shared["pixel_values"] = self._prepare_pixel_values(image)

            for point, mask in samples:
                yield {
                    **shared,
                    "point": point,
                    "mask": mask,
                }


class SamCollator:
    def __init__(self, *, precompute_backbone: bool) -> None:
        self._precompute_backbone = precompute_backbone

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        points = torch.tensor(
            [[[[x, y] for x, y in (item["point"],)]] for item in batch],
            dtype=torch.float32,
        )
        labels_pts = torch.ones((len(batch), 1, 1), dtype=torch.int64)
        masks = torch.from_numpy(np.stack([item["mask"] for item in batch])).float()

        out: dict[str, torch.Tensor | tuple[torch.Tensor, ...]] = {
            "input_points": points,
            "input_labels": labels_pts,
            "labels": masks,
        }

        if self._precompute_backbone:
            ihs_layers = zip(*(item["intermediate_hidden_states"] for item in batch))
            out["intermediate_hidden_states"] = tuple(
                torch.stack(list(layer_tensors), dim=0) for layer_tensors in ihs_layers
            )
        else:
            out["pixel_values"] = torch.cat([item["pixel_values"] for item in batch], dim=0)

        return out
