"""Florence training view over floorplan-to-json (no extraction column)."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

from datasets import Features, Image, Value, load_dataset

FEATURES = Features(
    {
        "stem": Value("string"),
        "plan_id": Value("string"),
        "variant": Value("int32"),
        "image": Image(),
        "florence": Value("string"),
    }
)

_COLUMNS = tuple(FEATURES.keys())


def load_from_hub(repo: str, *, revision: str = "main", cache_dir: Path | None = None):
    kwargs: dict[str, object] = {"split": "train", "revision": revision}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    token = os.environ.get("HF_TOKEN")
    if token:
        kwargs["token"] = token
    dataset = load_dataset(repo, **kwargs)
    missing = [col for col in _COLUMNS if col not in dataset.column_names]
    if missing:
        raise KeyError(f"Hub dataset missing columns: {missing}")
    return dataset.select_columns(list(_COLUMNS))


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
