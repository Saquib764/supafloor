"""Hub rows → grayscale floor plan input + five-class structure targets."""

from __future__ import annotations

import random
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Thread
from typing import Any, Iterator

import cv2
import numpy as np
import torch
from config import AugmentSettings, MaskChannels
from PIL import Image, ImageOps
from processing import get_adapter
from registry import load_dataset, load_rows_at
from torch.utils.data import IterableDataset


def prepare_image(image: Image.Image, size: int) -> Image.Image:
    """Grayscale → resize → invert (white walls on black background)."""
    gray = image.convert("L")
    if gray.size != (size, size):
        gray = gray.resize((size, size), Image.Resampling.LANCZOS)
    return ImageOps.invert(gray)


def prepare_image_array(image: Image.Image, size: int) -> np.ndarray:
    """Grayscale → resize → invert as (H, W) uint8."""
    return np.asarray(prepare_image(image, size), dtype=np.uint8)


def image_to_tensor(image: Image.Image | np.ndarray) -> torch.Tensor:
    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    else:
        array = image.astype(np.float32, copy=False) / 255.0
    if array.ndim != 2:
        raise ValueError(f"expected single-channel image, got shape {array.shape}")
    return torch.from_numpy(array).unsqueeze(0)


def _scaled_side(size: int, scale: float) -> int:
    return max(1, int(round(size * scale)))


def _resize_square_gray(gray: np.ndarray, size: int) -> np.ndarray:
    if gray.shape[0] == size and gray.shape[1] == size:
        return gray
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_LINEAR)


def _resize_square_masks(
    masks: np.ndarray,
    size: int,
    *,
    background_channel: int,
) -> np.ndarray:
    if masks.shape[-2] == size and masks.shape[-1] == size:
        return masks
    channels = []
    for channel_index in range(masks.shape[0]):
        channel = masks[channel_index]
        channels.append(
            cv2.resize(channel, (size, size), interpolation=cv2.INTER_NEAREST)
        )
    stacked = np.stack(channels, axis=0).astype(np.float32, copy=False)
    if background_channel >= 0:
        stacked[background_channel] = (stacked[background_channel] > 0.5).astype(np.float32)
    return stacked


def scale_crop_gray(gray: np.ndarray, size: int, scale: float) -> np.ndarray:
    gray = _resize_square_gray(gray, size)
    scaled_side = _scaled_side(size, scale)
    if scaled_side != size:
        gray = cv2.resize(gray, (scaled_side, scaled_side), interpolation=cv2.INTER_LINEAR)
    if scaled_side >= size:
        left = (scaled_side - size) // 2
        top = (scaled_side - size) // 2
        return gray[top : top + size, left : left + size]
    canvas = np.zeros((size, size), dtype=gray.dtype)
    paste = (size - scaled_side) // 2
    canvas[paste : paste + scaled_side, paste : paste + scaled_side] = gray
    return canvas


def scale_crop_masks(
    masks: np.ndarray,
    size: int,
    scale: float,
    *,
    background_channel: int,
) -> np.ndarray:
    if masks.ndim != 3:
        raise ValueError(f"masks must be (C, H, W), got {masks.shape}")
    masks = _resize_square_masks(masks, size, background_channel=background_channel)
    scaled_side = _scaled_side(size, scale)
    channels = []
    for channel_index in range(masks.shape[0]):
        channel = masks[channel_index]
        fill = 1.0 if channel_index == background_channel else 0.0
        if scaled_side != size:
            channel = cv2.resize(
                channel,
                (scaled_side, scaled_side),
                interpolation=cv2.INTER_NEAREST,
            )
        if scaled_side >= size:
            left = (scaled_side - size) // 2
            top = (scaled_side - size) // 2
            channel = channel[top : top + size, left : left + size]
        else:
            canvas = np.full((size, size), fill, dtype=np.float32)
            paste = (size - scaled_side) // 2
            canvas[paste : paste + scaled_side, paste : paste + scaled_side] = channel
            channel = canvas
        channels.append(channel.astype(np.float32, copy=False))
    return np.stack(channels, axis=0)


def apply_opacity(gray: np.ndarray, opacity: float) -> np.ndarray:
    clipped = float(np.clip(opacity, 0.0, 1.0))
    return np.clip(gray.astype(np.float32) * clipped, 0.0, 255.0).astype(np.uint8)


def augment_sample(
    image: Image.Image | np.ndarray,
    masks: np.ndarray,
    *,
    rng: random.Random,
    augment: AugmentSettings,
    background_channel: int,
    size: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(image, Image.Image):
        if size is None:
            size = image.size[0]
        gray = np.asarray(image.convert("L"), dtype=np.uint8)
    else:
        gray = image
        if size is None:
            size = int(gray.shape[0])
    scale = rng.uniform(augment.scale_min, augment.scale_max)
    opacity = rng.uniform(augment.opacity_min, augment.opacity_max)
    scaled_gray = scale_crop_gray(gray, size, scale)
    scaled_masks = scale_crop_masks(
        masks,
        size,
        scale,
        background_channel=background_channel,
    )
    return apply_opacity(scaled_gray, opacity), scaled_masks


def _select_hub_columns(dataset: Any, columns: tuple[str, ...]) -> Any:
    missing = [col for col in columns if col not in dataset.column_names]
    if missing:
        raise KeyError(f"Hub dataset missing columns: {missing}")
    return dataset.select_columns(list(columns))


def load_train_split(
    dataset: str,
    *,
    test_fraction: float,
    source: str = "auto",
    revision: str | None = None,
    cache_dir: Any = None,
) -> tuple[Any, int, int]:
    """Load train rows from local folder or Hub (materialized Dataset, indexed by row)."""
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError("data.test_fraction must be between 0 (inclusive) and 1")
    adapter = get_adapter(dataset)
    rows = _select_hub_columns(
        load_dataset(dataset, revision=revision, cache_dir=cache_dir, source=source),
        adapter.hub_columns,
    )
    total = len(rows)
    if total < 1:
        raise ValueError(f"dataset {dataset!r} has no rows")
    if test_fraction <= 0.0 or total < 2:
        return rows, total, 0
    train_pct = int(round((1.0 - test_fraction) * 100))
    if not 0 < train_pct < 100:
        raise ValueError("data.test_fraction must leave at least one train and test row")
    train_count = int(total * train_pct / 100)
    test_count = total - train_count
    return rows, train_count, test_count


def _plan_id(stem: str) -> str:
    if "_" not in stem:
        return stem
    plan_id, rest = stem.rsplit("_", 1)
    try:
        int(rest)
    except ValueError:
        return stem
    return plan_id


def load_viz_rows(
    dataset: str,
    train_count: int,
    *,
    count: int,
    source: str = "auto",
    seed: int | None = None,
    revision: str | None = None,
    cache_dir: Any = None,
) -> list[dict[str, Any]]:
    """Random test-split rows, one variant per floor plan."""
    if count < 1:
        return []
    adapter = get_adapter(dataset)
    rows = load_dataset(dataset, revision=revision, cache_dir=cache_dir, source=source)
    total = len(rows)
    test_indices = list(range(train_count, total))
    if not test_indices:
        return []
    stems = load_rows_at(rows, test_indices, columns=("stem",))
    by_plan: dict[str, list[int]] = {}
    for index, row in zip(test_indices, stems, strict=True):
        plan = _plan_id(str(row["stem"]))
        by_plan.setdefault(plan, []).append(index)
    rng = random.Random(seed)
    plans = list(by_plan)
    rng.shuffle(plans)
    picked = plans[: min(count, len(plans))]
    indices = [rng.choice(by_plan[plan]) for plan in picked]
    return load_rows_at(rows, indices, columns=adapter.hub_columns)


def load_training_sample(
    row: dict[str, Any],
    *,
    dataset_name: str,
    size: int,
    mask_channels: MaskChannels,
    image_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    adapter = get_adapter(dataset_name)
    image, targets = adapter.load_sample(
        row,
        size=size,
        channels=mask_channels,
        image_key=image_key,
    )
    return prepare_image_array(image, size), targets


@dataclass(frozen=True)
class _CachedSample:
    gray: np.ndarray
    masks: np.ndarray


def _cache_sample(
    row: dict[str, Any],
    *,
    adapter: Any,
    mask_channels: MaskChannels,
    image_key: str,
    cache_size: int,
) -> _CachedSample:
    image, targets = adapter.load_sample(
        row,
        size=cache_size,
        channels=mask_channels,
        image_key=image_key,
    )
    return _CachedSample(
        gray=prepare_image_array(image, cache_size),
        masks=targets.astype(np.float32, copy=False),
    )


class FloorplanDataset(IterableDataset):
    def __init__(
        self,
        dataset: Any,
        *,
        dataset_name: str,
        mask_channels: MaskChannels,
        input_sizes: tuple[int, ...],
        batch_size: int,
        num_rows: int,
        augment: AugmentSettings,
        cache_size: int,
        seed: int | None = None,
        prefetch_samples: int = 0,
    ) -> None:
        self._adapter = get_adapter(dataset_name)
        self._mask_channels = mask_channels
        self._input_sizes = input_sizes
        self._batch_size = max(1, batch_size)
        self._augment = augment
        self._seed = seed
        self._prefetch_samples = max(0, prefetch_samples)
        self.epoch = 0
        self._cache: list[_CachedSample] = []
        for dataset_index in range(num_rows):
            row = dataset[dataset_index]
            for image_key in self._adapter.image_keys:
                self._cache.append(
                    _cache_sample(
                        row,
                        adapter=self._adapter,
                        mask_channels=mask_channels,
                        image_key=image_key,
                        cache_size=cache_size,
                    )
                )

    def __len__(self) -> int:
        return len(self._cache)

    def _worker_indices(self, indices: list[int]) -> list[int]:
        info = torch.utils.data.get_worker_info()
        if info is None:
            return indices
        return indices[info.id :: info.num_workers]

    def _iter_samples(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        indices = list(range(len(self._cache)))
        rng = random.Random((self._seed or 0) + self.epoch)
        rng.shuffle(indices)
        indices = self._worker_indices(indices)
        current_size: int | None = None
        yields_in_batch = 0
        for cache_index in indices:
            sample = self._cache[cache_index]
            if yields_in_batch % self._batch_size == 0:
                current_size = rng.choice(self._input_sizes)
            gray, masks = augment_sample(
                sample.gray,
                sample.masks,
                rng=rng,
                augment=self._augment,
                background_channel=self._mask_channels.background,
                size=current_size,
            )
            yields_in_batch += 1
            yield image_to_tensor(gray), torch.from_numpy(masks)

    def _iter_prefetched(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        queue: Queue[tuple[torch.Tensor, torch.Tensor] | None] = Queue(
            maxsize=max(1, self._prefetch_samples // self._batch_size),
        )

        def producer() -> None:
            try:
                for item in self._iter_samples():
                    queue.put(item)
            finally:
                queue.put(None)

        thread = Thread(target=producer, daemon=True)
        thread.start()
        while True:
            try:
                item = queue.get(timeout=0.1)
            except Empty:
                if not thread.is_alive() and queue.empty():
                    break
                continue
            if item is None:
                break
            yield item

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        if self._prefetch_samples > 0:
            yield from self._iter_prefetched()
            return
        yield from self._iter_samples()


class FloorplanCollator:
    def __call__(self, batch: list[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, torch.Tensor]:
        images, masks = zip(*batch, strict=True)
        return {
            "images": torch.stack(list(images)),
            "masks": torch.stack(list(masks)),
        }
