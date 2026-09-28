"""Load RACE via dataset_manager; stream from local JSONL when available."""

from __future__ import annotations

import json
import random
from random import Random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from config import DecisionConfig, datasets_cache_dir
from datasets import Dataset
from formatting import FormattedBatch, FormattedTrainBatch, format_batch, format_train_batch
from model.decision_model import ModernBERTDecisionModel
from registry import _metadata, _schema, local_dataset_available, local_dataset_root
from torch.utils.data import Dataset as TorchDataset, IterableDataset


@dataclass(frozen=True)
class DecisionExample:
    state: str
    question: str
    options: list[str]
    answer_index: int


@dataclass(frozen=True)
class TrainRow:
    state: str
    question: str
    options: list[str]
    candidate: str
    label: float


def _shuffled_wrong_indices(example: DecisionExample, rng: Random) -> list[int]:
    wrong = [i for i in range(len(example.options)) if i != example.answer_index]
    order = wrong.copy()
    rng.shuffle(order)
    return order


def train_row_for_variation(
    example: DecisionExample,
    variation: int,
    *,
    wrong_order: list[int],
) -> TrainRow:
    if variation == 0:
        candidate_index = example.answer_index
    else:
        candidate_index = wrong_order[(variation - 1) % len(wrong_order)]
    candidate = example.options[candidate_index]
    label = 1.0 if candidate_index == example.answer_index else 0.0
    return TrainRow(
        state=example.state,
        question=example.question,
        options=example.options,
        candidate=candidate,
        label=label,
    )


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
) -> Iterator[DecisionExample]:
    schema = _schema(cfg.data.dataset)
    root = race_local_root(cfg)
    if shuffle:
        rows = schema.iter_split_rows_shuffled(root, split, seed=seed)
    else:
        rows = schema.iter_split_rows(root, split)
    for row in rows:
        yield race_row_to_example(row)


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


def _examples_from_hub_split(cfg: DecisionConfig, split: str) -> list[DecisionExample]:
    """Rows from ``load_race_hub_split`` are already registry FEATURES (state, answer_index)."""
    rows = load_race_hub_split(cfg, split=split)
    return [race_row_to_example(dict(rows[int(i)])) for i in range(len(rows))]


def load_race_split_examples(
    cfg: DecisionConfig,
    *,
    split: str,
    count: int,
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
        out.append(ex)
    return out


def load_race_preview_examples(cfg: DecisionConfig, count: int = 2) -> list[DecisionExample]:
    return load_race_split_examples(cfg, split=cfg.data.train_split, count=count)


EVAL_SAMPLES_FILENAME = "eval_samples.jsonl"
TEST_TRAIN_SAMPLES_FILENAME = "test_train_samples.jsonl"


def example_to_json(ex: DecisionExample) -> dict[str, object]:
    return {
        "state": ex.state,
        "question": ex.question,
        "options": ex.options,
        "answer_index": ex.answer_index,
    }


def example_from_json(row: dict[str, object]) -> DecisionExample:
    return race_row_to_example(row)


def write_eval_examples(path: Path, examples: list[DecisionExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(example_to_json(ex), ensure_ascii=False) for ex in examples)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def read_eval_examples(path: Path) -> list[DecisionExample]:
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[DecisionExample] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        out.append(example_from_json(json.loads(stripped)))
    return out


def _validation_example_pool(cfg: DecisionConfig) -> list[DecisionExample]:
    if use_local_race(cfg):
        return list(
            iter_race_local_examples(
                cfg,
                cfg.data.val_split,
                shuffle=False,
                seed=cfg.data.seed,
            )
        )
    return _examples_from_hub_split(cfg, cfg.data.val_split)


def pick_eval_examples_from_validation(
    cfg: DecisionConfig,
    *,
    count: int,
    seed: int,
) -> list[DecisionExample]:
    pool = _validation_example_pool(cfg)
    if not pool:
        return []
    rng = random.Random(seed)
    order = list(range(len(pool)))
    rng.shuffle(order)
    take = min(count, len(order))
    return [pool[i] for i in order[:take]]


def _train_example_pool(cfg: DecisionConfig) -> list[DecisionExample]:
    if use_local_race(cfg):
        return list(
            iter_race_local_examples(
                cfg,
                cfg.data.train_split,
                shuffle=False,
                seed=cfg.data.seed,
            )
        )
    return _examples_from_hub_split(cfg, cfg.data.train_split)


def pick_train_examples(
    cfg: DecisionConfig,
    *,
    count: int,
    seed: int,
) -> list[DecisionExample]:
    pool = _train_example_pool(cfg)
    if not pool:
        return []
    rng = random.Random(seed)
    order = list(range(len(pool)))
    rng.shuffle(order)
    take = min(count, len(order))
    return [pool[i] for i in order[:take]]


def ensure_eval_examples(cfg: DecisionConfig) -> tuple[list[DecisionExample], Path]:
    path = cfg.train.session_dir / EVAL_SAMPLES_FILENAME
    if path.is_file():
        return read_eval_examples(path), path
    picked = pick_eval_examples_from_validation(
        cfg,
        count=cfg.train.eval_samples,
        seed=cfg.train.eval_seed,
    )
    write_eval_examples(path, picked)
    return picked, path


def ensure_test_train_examples(cfg: DecisionConfig) -> tuple[list[DecisionExample], Path]:
    path = cfg.train.session_dir / TEST_TRAIN_SAMPLES_FILENAME
    if path.is_file():
        return read_eval_examples(path), path
    picked = pick_train_examples(
        cfg,
        count=cfg.train.test_samples,
        seed=cfg.data.seed,
    )
    write_eval_examples(path, picked)
    return picked, path


class ListDecisionDataset(TorchDataset):
    def __init__(self, examples: list[DecisionExample]) -> None:
        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> DecisionExample:
        return self._examples[int(index)]


class ExpandedMapTrainDataset(TorchDataset):
    """One dataset row per (example × variation); batch_size applies to these rows."""

    def __init__(
        self,
        base: TorchDataset,
        *,
        variations: int,
        seed: int,
    ) -> None:
        self._base = base
        self._variations = max(1, variations)
        self._seed = seed

    def __len__(self) -> int:
        return len(self._base) * self._variations

    def __getitem__(self, index: int) -> TrainRow:
        base_idx = int(index) // self._variations
        slot = int(index) % self._variations
        ex = self._base[base_idx]
        rng = Random(self._seed + base_idx)
        wrong_order = _shuffled_wrong_indices(ex, rng)
        return train_row_for_variation(ex, slot, wrong_order=wrong_order)


class ExpandedTrainStreamDataset(IterableDataset):
    def __init__(
        self,
        cfg: DecisionConfig,
        *,
        split: str,
        variations: int,
        shuffle: bool = False,
        seed: int = 0,
        epoch: int = 0,
    ) -> None:
        self._inner = RaceDecisionStreamDataset(
            cfg,
            split=split,
            shuffle=shuffle,
            seed=seed,
            epoch=epoch,
        )
        self._variations = max(1, variations)
        self._rng = Random(seed + epoch)

    @property
    def epoch(self) -> int:
        return self._inner.epoch

    @epoch.setter
    def epoch(self, value: int) -> None:
        self._inner.epoch = value
        self._rng = Random(self._inner.seed + value)

    def __iter__(self) -> Iterator[TrainRow]:
        for ex in self._inner:
            wrong_order = _shuffled_wrong_indices(ex, self._rng)
            for slot in range(self._variations):
                yield train_row_for_variation(ex, slot, wrong_order=wrong_order)


def load_race_decision_dataset(cfg: DecisionConfig, *, split: str) -> Dataset:
    if use_local_race(cfg):
        raise RuntimeError("use RaceDecisionStreamDataset for local RACE (streaming JSONL)")
    return load_race_hub_split(cfg, split=split)


class RaceDecisionDataset(TorchDataset):
    def __init__(self, rows: Dataset, cfg: DecisionConfig) -> None:
        self._rows = rows
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> DecisionExample:
        return race_row_to_example(self._rows[int(index)])


class RaceDecisionStreamDataset(IterableDataset):
    def __init__(
        self,
        cfg: DecisionConfig,
        *,
        split: str,
        shuffle: bool = False,
        seed: int = 0,
        epoch: int = 0,
    ) -> None:
        self.cfg = cfg
        self.split = split
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = epoch

    def __iter__(self) -> Iterator[DecisionExample]:
        epoch_seed = self.seed + self.epoch
        yield from iter_race_local_examples(
            self.cfg,
            self.split,
            shuffle=self.shuffle,
            seed=epoch_seed,
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
    formatted: FormattedTrainBatch
    labels: torch.Tensor


@dataclass
class EvalBatch:
    formatted: FormattedBatch
    labels: torch.Tensor


class TrainCollator:
    def __init__(self, model: ModernBERTDecisionModel) -> None:
        self.model = model

    def __call__(self, rows: list[TrainRow]) -> TrainBatch:
        sequences = [(r.state, r.question, r.options, r.candidate) for r in rows]
        formatted = format_train_batch(
            sequences,
            cfg=self.model.cfg,
            tokenizer=self.model.tokenizer,
            max_len=self.model.max_len,
        )
        labels = torch.tensor([r.label for r in rows], dtype=torch.float32)
        return TrainBatch(formatted=formatted, labels=labels)


class EvalCollator:
    def __init__(self, model: ModernBERTDecisionModel) -> None:
        self.model = model

    def __call__(self, examples: list[DecisionExample]) -> EvalBatch:
        requests = [example_to_request(ex) for ex in examples]
        formatted = format_batch(
            requests,
            cfg=self.model.cfg,
            tokenizer=self.model.tokenizer,
            max_len=self.model.max_len,
        )
        labels = torch.tensor([ex.answer_index for ex in examples], dtype=torch.long)
        return EvalBatch(formatted=formatted, labels=labels)
