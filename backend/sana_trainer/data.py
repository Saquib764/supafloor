"""Hub or local Hugging Face datasets with image + caption columns."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop

from config import Config, datasets_cache_dir


def load_split(cfg: Config) -> Any:
    name = cfg.data.dataset
    kwargs: dict[str, Any] = {"split": cfg.data.split}
    if cfg.data.revision:
        kwargs["revision"] = cfg.data.revision
    cache = datasets_cache_dir(cfg)
    path = Path(name).expanduser()
    if path.is_dir():
        return load_dataset(str(path), **kwargs, cache_dir=str(cache))
    return load_dataset(name, **kwargs, cache_dir=str(cache))


def _validate_columns(split: Any, cfg: Config) -> None:
    columns = split.column_names
    for col in (cfg.data.image_column, cfg.data.caption_column):
        if col not in columns:
            raise ValueError(
                f"data.{col} column {col!r} not in dataset columns: {', '.join(columns)}"
            )


class CaptionImageDataset(Dataset):
    def __init__(self, split: Any, cfg: Config) -> None:
        _validate_columns(split, cfg)
        self._split = split
        self._cfg = cfg
        size = cfg.resolution
        self._resize = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)
        self._center_crop = transforms.CenterCrop(size)
        self._random_crop = transforms.RandomCrop(size)
        self._flip = transforms.RandomHorizontalFlip(p=1.0)
        self._to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self._split)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self._split[index]
        image = row[self._cfg.data.image_column]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = exif_transpose(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = self._resize(image)
        if self._cfg.data.random_flip and random.random() < 0.5:
            image = self._flip(image)
        if self._cfg.data.center_crop:
            image = self._center_crop(image)
        else:
            crop_params = self._random_crop.get_params(image, (self._cfg.resolution, self._cfg.resolution))
            image = crop(image, *crop_params)
        caption = row[self._cfg.data.caption_column]
        if caption is None or (isinstance(caption, str) and not caption.strip()):
            raise ValueError(f"empty caption at dataset index {index}")
        return {
            "pixel_values": self._to_tensor(image),
            "caption": str(caption).strip(),
        }


def collate_batch(examples: list[dict[str, Any]]) -> dict[str, Any]:
    pixels = torch.stack([ex["pixel_values"] for ex in examples])
    return {"pixel_values": pixels, "captions": [ex["caption"] for ex in examples]}
