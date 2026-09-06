"""Hub image → simple_image stream, both as 1×image_size×image_size float tensors."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from PIL import Image
from registry import load_hub_dataset
from torch.utils.data import IterableDataset

_HUB_RETRIES = 5
_SHUFFLE_BUFFER = 256


def prepare_image(image: Image.Image | np.ndarray, size: int) -> torch.Tensor:
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    image = image.convert("L").resize((size, size), Image.Resampling.BILINEAR)
    array = np.clip(np.asarray(image, dtype=np.float32) / 255.0, 0.0, 1.0)
    return torch.from_numpy(array).unsqueeze(0)


def _is_timeout(exc: BaseException) -> bool:
    return "timeout" in type(exc).__name__.lower() or "timed out" in str(exc).lower()


def load_rows(dataset: str, *, revision: str | None = None, cache_dir: Path | None = None):
    for attempt in range(1, _HUB_RETRIES + 1):
        try:
            return load_hub_dataset(dataset, revision=revision, cache_dir=cache_dir)
        except Exception as exc:
            if not _is_timeout(exc) or attempt == _HUB_RETRIES:
                raise
            wait = min(2 ** (attempt - 1), 16)
            print(f"Hub timeout ({attempt}/{_HUB_RETRIES}), retry in {wait}s…", flush=True)
            time.sleep(wait)


def hub_row_count(rows: Any) -> int:
    splits = getattr(rows.info, "splits", None) or {}
    train = splits.get("train") if hasattr(splits, "get") else None
    if train is None or not train.num_examples:
        raise FileNotFoundError("Hub stream has no train example count")
    return int(train.num_examples)


class FloorplanVaeDataset(IterableDataset):
    def __init__(
        self,
        rows: Any,
        image_size: int,
        *,
        num_rows: int,
        seed: int | None = None,
    ) -> None:
        self._rows = rows
        self._image_size = image_size
        self._num_rows = num_rows
        self._seed = seed or 0
        self.epoch = 0

    def __len__(self) -> int:
        return self._num_rows * 2

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        stream = self._rows.shuffle(seed=self._seed + self.epoch, buffer_size=_SHUFFLE_BUFFER)
        for row in stream:
            source = prepare_image(row["image"], self._image_size)
            target = prepare_image(row["simple_image"], self._image_size)
            yield source, target
            yield target, target
