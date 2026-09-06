"""VAE view over floorplan-to-json: Hub stream of image → simple_image (no local folders)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from datasets import Features, Image, Value, load_dataset, load_dataset_builder

FEATURES = Features(
    {
        "stem": Value("string"),
        "plan_id": Value("string"),
        "variant": Value("int32"),
        "image": Image(),
        "simple_image": Image(),
    }
)

_TRAIN_COLUMNS = ("image", "simple_image")


def _hub_kwargs(repo: str, *, revision: str, cache_dir: Path | None) -> dict[str, object]:
    if "/" not in repo or Path(repo).is_dir():
        raise ValueError(f"floorplan-vae loads from the Hub only, not a local path: {repo}")
    kwargs: dict[str, object] = {"revision": revision}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    token = os.environ.get("HF_TOKEN")
    if token:
        kwargs["token"] = token
    return kwargs


def hub_num_rows(repo: str, *, revision: str = "main", cache_dir: Path | None = None) -> int:
    builder = load_dataset_builder(repo, **_hub_kwargs(repo, revision=revision, cache_dir=cache_dir))
    splits = builder.info.splits or {}
    if "train" not in splits or not splits["train"].num_examples:
        raise FileNotFoundError(f"Hub dataset {repo!r} has no train split size")
    return int(splits["train"].num_examples)


def load_from_hub(repo: str, *, revision: str = "main", cache_dir: Path | None = None):
    kwargs = _hub_kwargs(repo, revision=revision, cache_dir=cache_dir)
    dataset = load_dataset(repo, split="train", streaming=True, **kwargs)
    missing = [col for col in _TRAIN_COLUMNS if col not in dataset.column_names]
    if missing:
        raise KeyError(f"Hub dataset missing columns: {missing}")
    return dataset.select_columns(list(_TRAIN_COLUMNS))


def load_samples_from_hub(
    repo: str,
    *,
    count: int,
    seed: int | None = None,
    revision: str = "main",
    cache_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], list[int], int]:
    dataset = load_from_hub(repo, revision=revision, cache_dir=cache_dir)
    num_rows = hub_num_rows(repo, revision=revision, cache_dir=cache_dir)
    if num_rows == 0:
        raise FileNotFoundError(f"No rows in Hub dataset {repo!r}")
    stream = dataset.shuffle(seed=seed, buffer_size=max(count * 8, 64)) if seed is not None else dataset
    rows: list[dict[str, Any]] = []
    for row in stream:
        rows.append(row)
        if len(rows) >= min(count, num_rows):
            break
    return rows, list(range(len(rows))), num_rows
