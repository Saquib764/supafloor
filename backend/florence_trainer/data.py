"""Hub rows, training/infer sample builders, and Florence collator."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

from datasets import Dataset as HubDataset
from PIL import Image
from config import _load_dotenv as _load_florence_dotenv
from loc import format_loc_point, pixel_to_loc
from registry import load_dotenv, load_hub_dataset, load_hub_samples
from torch.utils.data import Dataset


def prepare_image(image: Image.Image, size: int) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.LANCZOS)
    return image


def load_rows(dataset: str, *, revision: str | None = None, cache_dir: Path | None = None) -> HubDataset:
    _load_florence_dotenv()
    load_dotenv()
    return load_hub_dataset(dataset, revision=revision, cache_dir=cache_dir)


def load_sample_rows(
    dataset: str,
    *,
    count: int,
    seed: int | None = None,
    revision: str | None = None,
    cache_dir: Path | None = None,
) -> tuple[list[dict[str, object]], list[int], int]:
    _load_florence_dotenv()
    load_dotenv()
    return load_hub_samples(
        dataset,
        count=count,
        seed=seed,
        revision=revision,
        cache_dir=cache_dir,
    )


def _point_in_polygon(x: float, y: float, ring: list[tuple[float, float]]) -> bool:
    inside = False
    j = len(ring) - 1
    for i, (xi, yi) in enumerate(ring):
        xj, yj = ring[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _polygon_centroid(ring: list[tuple[float, float]]) -> tuple[float, float]:
    if not ring:
        return 0.0, 0.0
    return sum(x for x, _ in ring) / len(ring), sum(y for _, y in ring) / len(ring)


def _point_inside_any(x: float, y: float, rings: list[list[tuple[float, float]]]) -> bool:
    return any(_point_in_polygon(x, y, ring) for ring in rings)


def polygon_to_canvas_ring(
    polygon: list[list[int | float]],
    *,
    image_size: int,
    canvas_size: int,
) -> list[tuple[float, float]]:
    scale = canvas_size / image_size
    return [(float(x) * scale, float(y) * scale) for x, y in polygon]


def random_point_in_polygon(
    polygon: list[dict[str, Any]],
    rng: random.Random,
) -> tuple[float, float]:
    ring = [(float(point["x"]), float(point["y"])) for point in polygon]
    if len(ring) < 3:
        raise ValueError("polygon needs at least 3 vertices")

    minx = min(x for x, _ in ring)
    maxx = max(x for x, _ in ring)
    miny = min(y for _, y in ring)
    maxy = max(y for _, y in ring)
    for _ in range(100):
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        if _point_in_polygon(x, y, ring):
            return x, y

    return _polygon_centroid(ring)


def random_point_near_center(
    rng: random.Random,
    canvas_size: int,
    *,
    spread_ratio: float = 0.08,
) -> tuple[float, float]:
    center = canvas_size / 2.0
    spread = canvas_size * spread_ratio
    return (
        rng.uniform(center - spread, center + spread),
        rng.uniform(center - spread, center + spread),
    )


def random_point_outside_polygon(
    ring: list[tuple[float, float]],
    rng: random.Random,
    canvas_size: int,
) -> tuple[float, float] | None:
    if len(ring) < 3:
        return None
    cx, cy = _polygon_centroid(ring)
    max_r = max(math.hypot(x - cx, y - cy) for x, y in ring)
    for _ in range(80):
        angle = rng.uniform(0.0, 2.0 * math.pi)
        dist = max_r + rng.uniform(16.0, 96.0)
        x = cx + math.cos(angle) * dist
        y = cy + math.sin(angle) * dist
        if (
            0.0 <= x < canvas_size
            and 0.0 <= y < canvas_size
            and not _point_in_polygon(x, y, ring)
        ):
            return x, y
    return random_point_outside_polygons([ring], rng, canvas_size)


def random_point_outside_polygons(
    rings: list[list[tuple[float, float]]],
    rng: random.Random,
    canvas_size: int,
) -> tuple[float, float] | None:
    valid = [ring for ring in rings if len(ring) >= 3]
    if not valid:
        return random_point_near_center(rng, canvas_size)

    for _ in range(300):
        x = rng.uniform(0.0, canvas_size)
        y = rng.uniform(0.0, canvas_size)
        if not _point_inside_any(x, y, valid):
            return x, y
    return None


def room_display_name(room: dict[str, Any]) -> str:
    if (room.get("label") or {}).get("is_label_disabled"):
        return "unlabeled"
    name = str(room.get("name") or "").strip()
    return name or "unlabeled"


def format_training_label(
    room: dict[str, Any],
    *,
    canvas_size: int,
    bins: int,
) -> str:
    polygon = room.get("polygon") or []
    vertex_locs = "".join(
        format_loc_point(float(point["x"]), float(point["y"]), canvas_size=canvas_size, bins=bins)
        for point in polygon
    )
    return f"<poly>{vertex_locs}</poly>\nRoom name: {room_display_name(room)}"


def format_training_prompt(
    task_prefix: str,
    x: float,
    y: float,
    *,
    canvas_size: int,
    bins: int,
) -> str:
    point = format_loc_point(x, y, canvas_size=canvas_size, bins=bins)
    return f"{task_prefix}room containing {point}"


def format_infer_prompt(task_prefix: str, loc_x: int, loc_y: int) -> str:
    return f"{task_prefix}room containing <loc_{int(loc_x)}><loc_{int(loc_y)}>"


def canvas_point_to_loc(
    x: float,
    y: float,
    *,
    canvas_size: int,
    bins: int,
) -> tuple[int, int]:
    return (
        pixel_to_loc(x, canvas_size=canvas_size, bins=bins),
        pixel_to_loc(y, canvas_size=canvas_size, bins=bins),
    )


def build_referring_sample(
    extraction: dict[str, Any],
    index: int,
    *,
    task_prefix: str,
    canvas_size: int,
    bins: int,
    seed: int | None,
) -> tuple[str, str]:
    rooms = [
        room
        for room in extraction.get("rooms") or []
        if len(room.get("polygon") or []) >= 3
    ]
    if not rooms:
        raise ValueError("extraction has no valid room polygons")

    rng = random.Random((seed or 0) * 1_000_003 + index)
    room = rng.choice(rooms)
    px, py = random_point_in_polygon(room["polygon"], rng)
    prompt = format_training_prompt(
        task_prefix,
        px,
        py,
        canvas_size=canvas_size,
        bins=bins,
    )
    label = format_training_label(room, canvas_size=canvas_size, bins=bins)
    return prompt, label


def sample_infer_point(
    iteration: int,
    canvas_rings: list[list[tuple[float, float]]],
    rng: random.Random,
    canvas_size: int,
) -> tuple[float, float] | None:
    if iteration == 0:
        return random_point_near_center(rng, canvas_size)
    if iteration == 1 and len(canvas_rings) == 1:
        return random_point_outside_polygon(canvas_rings[0], rng, canvas_size)
    return random_point_outside_polygons(canvas_rings, rng, canvas_size)


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
    return _point_inside_any(cx * scale, cy * scale, canvas_rings)


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
        task_prompt: str,
        image_size: int,
        *,
        loc_bins: int,
        canvas_size: int,
        seed: int | None = None,
    ) -> None:
        self._rows = rows
        self._task_prompt = task_prompt
        self._image_size = image_size
        self._loc_bins = loc_bins
        self._canvas_size = canvas_size
        self._seed = seed

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self._rows[index]
        extraction = json.loads(str(row["extraction"]))
        prompt, label = build_referring_sample(
            extraction,
            index,
            task_prefix=self._task_prompt,
            canvas_size=self._canvas_size,
            bins=self._loc_bins,
            seed=self._seed,
        )
        return {
            "image": prepare_image(row["image"], self._image_size),
            "prompt": prompt,
            "label": label,
        }


class FlorenceCollator:
    def __init__(self, processor: object, max_label_tokens: int) -> None:
        self._processor = processor
        self._max_label_tokens = max_label_tokens

    def __call__(self, batch: list[dict[str, object]]) -> dict[str, object]:
        inputs = self._processor(
            text=[item["prompt"] for item in batch],
            images=[item["image"] for item in batch],
            return_tensors="pt",
            padding=True,
        )
        labels = self._processor.tokenizer(
            [item["label"] for item in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_label_tokens,
        ).input_ids
        labels[labels == self._processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        return inputs
