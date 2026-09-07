"""Manual floorplan Hub schema: one annotated sample per stem."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any
from uuid import uuid4

from datasets import Dataset, Features, Image, Value, load_dataset

JPEG_QUALITY = 95
IMAGE_EXT = ".jpg"

FEATURES = Features(
    {
        "stem": Value("string"),
        "plan_id": Value("string"),
        "variant": Value("int32"),
        "image": Image(),
        "extraction": Value("string"),
        "simple_image": Image(),
    }
)


def _iter_local_rows(src: Path):
    output_dir = src / "output"
    for image_path in sorted((src / "input").glob(f"*{IMAGE_EXT}")):
        stem = image_path.stem
        json_path = output_dir / f"{stem}.json"
        simple_path = src / "output2" / f"{stem}{IMAGE_EXT}"
        if not json_path.is_file() or not simple_path.is_file():
            continue
        yield {
            "stem": stem,
            "plan_id": uuid4().hex[:12],
            "variant": 0,
            "image": str(image_path),
            "extraction": json_path.read_text(encoding="utf-8"),
            "simple_image": str(simple_path),
        }


def build_from_local(local_root: Path) -> Dataset:
    local_root = Path(local_root)
    if not (local_root / "input").is_dir():
        raise FileNotFoundError(f"Missing input/ under {local_root}")
    dataset = Dataset.from_generator(
        _iter_local_rows,
        gen_kwargs={"src": local_root},
        features=FEATURES,
    )
    if len(dataset) == 0:
        raise FileNotFoundError(f"No complete input/output pairs under {local_root}")
    return dataset


def load_from_hub(repo: str, *, revision: str = "main", cache_dir: Path | None = None) -> Dataset:
    kwargs: dict[str, object] = {"split": "train", "revision": revision}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    token = os.environ.get("HF_TOKEN")
    if token:
        kwargs["token"] = token
    dataset = load_dataset(repo, **kwargs)
    columns = list(FEATURES.keys())
    missing = [col for col in columns if col not in dataset.column_names]
    if missing:
        raise KeyError(f"Hub dataset missing columns: {missing}")
    return dataset.select_columns(columns)


def load_samples_from_hub(
    repo: str,
    *,
    count: int,
    seed: int | None = None,
    revision: str = "main",
    cache_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], list[int], int]:
    dataset = load_from_hub(repo, revision=revision, cache_dir=cache_dir)
    num_rows = len(dataset)
    if num_rows == 0:
        raise FileNotFoundError(f"No rows in Hub dataset {repo!r}")

    rng = random.Random(seed)
    indices = (
        list(range(num_rows))
        if count >= num_rows
        else sorted(rng.sample(range(num_rows), count))
    )
    rows = [dataset[index] for index in indices]
    return rows, indices, num_rows


def write_local_folders(dataset: Dataset, target: Path) -> int:
    target = Path(target)
    input_dir = target / "input"
    output_dir = target / "output"
    output2_dir = target / "output2"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    output2_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for row in dataset:
        stem = str(row["stem"])
        row["image"].convert("RGB").save(
            input_dir / f"{stem}{IMAGE_EXT}", format="JPEG", quality=JPEG_QUALITY
        )
        (output_dir / f"{stem}.json").write_text(row["extraction"], encoding="utf-8")
        row["simple_image"].convert("RGB").save(
            output2_dir / f"{stem}{IMAGE_EXT}", format="JPEG", quality=JPEG_QUALITY
        )
        written += 1
    return written
