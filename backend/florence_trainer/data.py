"""Hub rows via dataset_manager, plan-id split, and Florence collator."""

from __future__ import annotations

import random
from pathlib import Path

from datasets import Dataset as HubDataset
from PIL import Image
from config import _load_dotenv as _load_florence_dotenv
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


def split_by_plan_id(
    rows: HubDataset,
    *,
    train_ratio: float,
    seed: int,
) -> tuple[HubDataset, HubDataset]:
    plan_ids_col = list(rows["plan_id"])
    plan_ids = sorted(set(plan_ids_col))
    rng = random.Random(seed)
    rng.shuffle(plan_ids)
    n_train = int(len(plan_ids) * train_ratio)
    if len(plan_ids) > 1:
        n_train = min(max(n_train, 1), len(plan_ids) - 1)
    train_ids = set(plan_ids[:n_train])
    train_idx = [i for i, plan_id in enumerate(plan_ids_col) if plan_id in train_ids]
    val_idx = [i for i, plan_id in enumerate(plan_ids_col) if plan_id not in train_ids]
    return rows.select(train_idx), rows.select(val_idx)


class FloorplanFlorenceDataset(Dataset):
    def __init__(self, rows: HubDataset, task_prompt: str, image_size: int) -> None:
        self._rows = rows
        self._task_prompt = task_prompt
        self._image_size = image_size

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self._rows[index]
        return {
            "image": prepare_image(row["image"], self._image_size),
            "prompt": self._task_prompt,
            "label": row["florence"],
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
