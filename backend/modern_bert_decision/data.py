"""Load RACE via dataset_manager; stream from local JSONL when available."""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch
from config import DecisionConfig, datasets_cache_dir
from datasets import Dataset
from formatting import FormattedBatch, format_batch
from model.decision_model import ModernBERTDecisionModel
from registry import _metadata, _schema, local_dataset_available, local_dataset_root
from torch.utils.data import Dataset as TorchDataset, IterableDataset


@dataclass(frozen=True)
class DecisionExample:
    state: str
    question: str
    options: list[str]
    answer_index: int


def race_row_to_example(row: dict[str, Any]) -> DecisionExample:
    state = str(row["state"]).strip()
    question = str(row["question"]).strip()
    options = [str(o).strip() for o in row["options"]]
    if not state or not question:
        raise ValueError("RACE row missing state or question")
    if len(options) < 2:
        raise ValueError("RACE row must have at least two options")
    if any(not opt for opt in options):
        raise ValueError("RACE row has blank option text")
    answer_index = int(row["answer_index"])
    if not 0 <= answer_index < len(options):
        raise ValueError(f"answer_index {answer_index} out of range for {len(options)} options")
    return DecisionExample(
        state=state,
        question=question,
        options=options,
        answer_index=answer_index,
    )


def _shuffle_options(options: list[str], answer_index: int, rng: random.Random) -> tuple[list[str], int]:
    order = list(range(len(options)))
    rng.shuffle(order)
    shuffled = [options[i] for i in order]
    new_answer = order.index(answer_index)
    return shuffled, new_answer


def augment_training_example(
    example: DecisionExample,
    rng: random.Random,
    cfg: DecisionConfig,
) -> DecisionExample:
    options = list(example.options)
    answer_index = example.answer_index
    none_label = cfg.data.none_of_these_option

    if len(options) > 2 and rng.random() < cfg.data.option_subset_prob:
        keep_count = rng.randint(2, len(options) - 1)
        kept = sorted(rng.sample(range(len(options)), keep_count))
        options = [options[i] for i in kept]
        if answer_index in kept:
            answer_index = kept.index(answer_index)
        else:
            options.append(none_label)
            answer_index = len(options) - 1

    options, answer_index = _shuffle_options(options, answer_index, rng)
    return DecisionExample(
        state=example.state,
        question=example.question,
        options=options,
        answer_index=answer_index,
    )


def example_to_request(example: DecisionExample) -> dict[str, object]:
    return {
        "state": example.state,
        "question": example.question,
        "options": example.options,
    }


def use_local_race(cfg: DecisionConfig) -> bool:
    if cfg.data.source == "hub":
        return False
    if cfg.data.source == "local":
        if not local_dataset_available(cfg.data.dataset):
            raise FileNotFoundError(
                f"data.source is local but {cfg.data.dataset!r} is not on disk "
                f"(run: uv run modern-bert-decision download-data)"
            )
        return True
    return local_dataset_available(cfg.data.dataset)


def race_local_root(cfg: DecisionConfig):
    root = local_dataset_root(cfg.data.dataset)
    if root is None:
        raise FileNotFoundError(f"dataset {cfg.data.dataset!r} has no local_root in metadata")
    return root


def race_split_row_count(cfg: DecisionConfig, split: str) -> int:
    schema = _schema(cfg.data.dataset)
    return schema.count_split_rows(race_local_root(cfg), split)


def iter_race_local_examples(
    cfg: DecisionConfig,
    split: str,
    *,
    shuffle: bool = False,
    seed: int = 0,
    augment: bool = False,
) -> Iterator[DecisionExample]:
    schema = _schema(cfg.data.dataset)
    root = race_local_root(cfg)
    if shuffle:
        rows = schema.iter_split_rows_shuffled(root, split, seed=seed)
    else:
        rows = schema.iter_split_rows(root, split)
    rng = random.Random(seed) if augment else None
    for row in rows:
        ex = race_row_to_example(row)
        if augment and rng is not None:
            ex = augment_training_example(ex, rng, cfg)
        yield ex


def load_race_hub_split(cfg: DecisionConfig, *, split: str) -> Dataset:
    meta = _metadata(cfg.data.dataset)
    schema = _schema(cfg.data.dataset)
    rev = meta.get("revision")
    return schema.load_from_hub(
        meta["hf_repo"],
        revision=rev,
        cache_dir=datasets_cache_dir(cfg),
        split=split,
        subset=cfg.data.race_subset,
    )


def load_race_split_examples(
    cfg: DecisionConfig,
    *,
    split: str,
    count: int,
    augment: bool = False,
) -> list[DecisionExample]:
    if count < 1:
        return []
    if use_local_race(cfg):
        out: list[DecisionExample] = []
        for ex in iter_race_local_examples(
            cfg,
            split,
            shuffle=False,
            seed=cfg.data.seed,
            augment=augment,
        ):
            out.append(ex)
            if len(out) >= count:
                break
        return out
    meta = _metadata(cfg.data.dataset)
    schema = _schema(cfg.data.dataset)
    cache = datasets_cache_dir(cfg)
    from datasets import load_dataset as hf_load_dataset

    rev = meta.get("revision")
    kwargs: dict[str, object] = {"cache_dir": str(cache)}
    if rev and str(rev).lower() not in ("null", "none", ""):
        kwargs["revision"] = rev
    split_slice = f"{split}[:{count}]"
    raw = hf_load_dataset(meta["hf_repo"], cfg.data.race_subset, split=split_slice, **kwargs)
    out: list[DecisionExample] = []
    for index in range(len(raw)):
        ex = race_row_to_example(schema.row_from_hf(dict(raw[int(index)])))
        if augment:
            ex = augment_training_example(ex, random.Random(cfg.data.seed + index), cfg)
        out.append(ex)
    return out


def load_race_preview_examples(
    cfg: DecisionConfig,
    count: int = 2,
    *,
    augment: bool = False,
) -> list[DecisionExample]:
    return load_race_split_examples(
        cfg,
        split=cfg.data.train_split,
        count=count,
        augment=augment,
    )


def load_race_eval_examples(cfg: DecisionConfig, *, count: int) -> list[DecisionExample]:
    return load_race_split_examples(
        cfg,
        split=cfg.data.val_split,
        count=count,
        augment=False,
    )


class ListDecisionDataset(TorchDataset):
    def __init__(self, examples: list[DecisionExample]) -> None:
        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> DecisionExample:
        return self._examples[int(index)]


def load_race_decision_dataset(cfg: DecisionConfig, *, split: str) -> Dataset:
    if use_local_race(cfg):
        raise RuntimeError("use RaceDecisionStreamDataset for local RACE (streaming JSONL)")
    return load_race_hub_split(cfg, split=split)


class RaceDecisionDataset(TorchDataset):
    def __init__(self, rows: Dataset, cfg: DecisionConfig, *, augment: bool = False) -> None:
        self._rows = rows
        self.cfg = cfg
        self.augment = augment

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> DecisionExample:
        ex = race_row_to_example(self._rows[int(index)])
        if self.augment:
            ex = augment_training_example(
                ex,
                random.Random(self.cfg.data.seed + int(index)),
                self.cfg,
            )
        return ex


class RaceDecisionStreamDataset(IterableDataset):
    def __init__(
        self,
        cfg: DecisionConfig,
        *,
        split: str,
        shuffle: bool = False,
        seed: int = 0,
        epoch: int = 0,
        augment: bool = False,
    ) -> None:
        self.cfg = cfg
        self.split = split
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = epoch
        self.augment = augment

    def __iter__(self) -> Iterator[DecisionExample]:
        epoch_seed = self.seed + self.epoch
        yield from iter_race_local_examples(
            self.cfg,
            self.split,
            shuffle=self.shuffle,
            seed=epoch_seed,
            augment=self.augment,
        )


def download_race_local(cfg: DecisionConfig, *, force: bool = False) -> Path:
    from registry import download

    return download(
        cfg.data.dataset,
        cache_dir=datasets_cache_dir(cfg),
        force=force,
    )


@dataclass
class TrainBatch:
    formatted: FormattedBatch
    labels: torch.Tensor


class DecisionCollator:
    def __init__(self, model: ModernBERTDecisionModel) -> None:
        self.model = model

    def __call__(self, examples: list[DecisionExample]) -> TrainBatch:
        requests = [example_to_request(ex) for ex in examples]
        formatted = format_batch(
            requests,
            tokenizer=self.model.tokenizer,
            markers=self.model.markers,
            max_len=self.model.max_len,
        )
        labels = torch.tensor([ex.answer_index for ex in examples], dtype=torch.long)
        return TrainBatch(formatted=formatted, labels=labels)
