"""Load decision datasets via dataset_manager; stream from local JSONL when available."""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
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


def registry_row_to_example(row: dict[str, Any]) -> DecisionExample:
    state = str(row["state"]).strip()
    question = str(row["question"]).strip()
    options = [str(o).strip() for o in row["options"]]
    if not state or not question:
        raise ValueError("row missing state or question")
    if len(options) < 2:
        raise ValueError("row must have at least two options")
    if any(not opt for opt in options):
        raise ValueError("row has blank option text")
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


def use_local_dataset(cfg: DecisionConfig) -> bool:
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


def dataset_local_root(cfg: DecisionConfig) -> Path:
    root = local_dataset_root(cfg.data.dataset)
    if root is None:
        raise FileNotFoundError(f"dataset {cfg.data.dataset!r} has no local_root in metadata")
    return root


def dataset_split_row_count(cfg: DecisionConfig, split: str) -> int:
    schema = _schema(cfg.data.dataset)
    return schema.count_split_rows(dataset_local_root(cfg), split)


def iter_local_examples(
    cfg: DecisionConfig,
    split: str,
    *,
    shuffle: bool = False,
    seed: int = 0,
) -> Iterator[DecisionExample]:
    schema = _schema(cfg.data.dataset)
    root = dataset_local_root(cfg)
    if shuffle:
        rows = schema.iter_split_rows_shuffled(root, split, seed=seed)
    else:
        rows = schema.iter_split_rows(root, split)
    for row in rows:
        yield registry_row_to_example(row)


def _hub_load_extra(cfg: DecisionConfig, meta: dict[str, object]) -> dict[str, object]:
    hub_config = meta.get("hub_config")
    if hub_config:
        return {"hub_config": str(hub_config)}
    return {"subset": cfg.data.race_subset}


def load_hub_split(cfg: DecisionConfig, *, split: str) -> Dataset:
    meta = _metadata(cfg.data.dataset)
    schema = _schema(cfg.data.dataset)
    rev = meta.get("revision")
    return schema.load_from_hub(
        meta["hf_repo"],
        revision=rev,
        cache_dir=datasets_cache_dir(cfg),
        split=split,
        **_hub_load_extra(cfg, meta),
    )


def _examples_from_hub_split(cfg: DecisionConfig, split: str) -> list[DecisionExample]:
    rows = load_hub_split(cfg, split=split)
    return [registry_row_to_example(dict(rows[int(i)])) for i in range(len(rows))]


def _hf_hub_kwargs(cfg: DecisionConfig) -> tuple[dict[str, object], Any, Any]:
    meta = _metadata(cfg.data.dataset)
    schema = _schema(cfg.data.dataset)
    kwargs: dict[str, object] = {"cache_dir": str(datasets_cache_dir(cfg))}
    rev = meta.get("revision")
    if rev and str(rev).lower() not in ("null", "none", ""):
        kwargs["revision"] = rev
    return kwargs, meta, schema


def load_split_examples(
    cfg: DecisionConfig,
    *,
    split: str,
    count: int,
) -> list[DecisionExample]:
    if count < 1:
        return []
    if use_local_dataset(cfg):
        out: list[DecisionExample] = []
        for ex in iter_local_examples(
            cfg,
            split,
            shuffle=False,
            seed=cfg.data.seed,
        ):
            out.append(ex)
            if len(out) >= count:
                break
        return out
    kwargs, meta, schema = _hf_hub_kwargs(cfg)
    from datasets import load_dataset as hf_load_dataset

    hub_config = meta.get("hub_config")
    if hub_config:
        raw = hf_load_dataset(
            meta["hf_repo"],
            str(hub_config),
            split=f"{split}[:{count}]",
            **kwargs,
        )
    else:
        raw = hf_load_dataset(
            meta["hf_repo"],
            cfg.data.race_subset,
            split=f"{split}[:{count}]",
            **kwargs,
        )
    out: list[DecisionExample] = []
    for index in range(len(raw)):
        ex = registry_row_to_example(schema.row_from_hf(dict(raw[int(index)])))
        out.append(ex)
    return out


def load_preview_examples(cfg: DecisionConfig, count: int = 2) -> list[DecisionExample]:
    return load_split_examples(cfg, split=cfg.data.train_split, count=count)


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
    return registry_row_to_example(row)


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
    if use_local_dataset(cfg):
        return list(
            iter_local_examples(
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


def pick_train_examples(
    cfg: DecisionConfig,
    *,
    count: int,
    seed: int,
) -> list[DecisionExample]:
    return pick_random_examples(
        cfg,
        split=cfg.data.train_split,
        count=count,
        seed=seed,
    )


def pick_random_examples(
    cfg: DecisionConfig,
    *,
    split: str,
    count: int,
    seed: int,
) -> list[DecisionExample]:
    if count < 1:
        return []
    if use_local_dataset(cfg):
        out: list[DecisionExample] = []
        for ex in iter_local_examples(
            cfg,
            split,
            shuffle=True,
            seed=seed,
        ):
            out.append(ex)
            if len(out) >= count:
                break
        return out
    pool = _examples_from_hub_split(cfg, split)
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


def ensure_test_train_examples(
    cfg: DecisionConfig,
    *,
    refresh: bool = False,
) -> tuple[list[DecisionExample], Path]:
    path = cfg.train.session_dir / TEST_TRAIN_SAMPLES_FILENAME
    if refresh and path.is_file():
        path.unlink()
    if path.is_file():
        return read_eval_examples(path), path
    picked = pick_train_examples(
        cfg,
        count=cfg.train.test_samples,
        seed=cfg.data.seed,
    )
    if not picked:
        raise FileNotFoundError(
            "train.test_mode: no training examples available "
            f"(split {cfg.data.train_split!r}); run download-data or fix data.source"
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


def load_decision_hub_dataset(cfg: DecisionConfig, *, split: str) -> Dataset:
    if use_local_dataset(cfg):
        raise RuntimeError("use DecisionStreamDataset for local JSONL (streaming)")
    return load_hub_split(cfg, split=split)


class RegistryDecisionDataset(TorchDataset):
    def __init__(self, rows: Dataset, cfg: DecisionConfig) -> None:
        self._rows = rows
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> DecisionExample:
        return registry_row_to_example(self._rows[int(index)])


class DecisionStreamDataset(IterableDataset):
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
        yield from iter_local_examples(
            self.cfg,
            self.split,
            shuffle=self.shuffle,
            seed=epoch_seed,
        )


def download_dataset_local(cfg: DecisionConfig, *, force: bool = False) -> Path:
    from registry import download

    return download(
        cfg.data.dataset,
        cache_dir=datasets_cache_dir(cfg),
        force=force,
    )


@dataclass
class DecisionBatch:
    formatted: FormattedBatch
    labels: torch.Tensor


class TrainCollator:
    def __init__(self, model: ModernBERTDecisionModel) -> None:
        self.model = model

    def __call__(self, examples: list[DecisionExample]) -> DecisionBatch:
        requests = [example_to_request(ex) for ex in examples]
        formatted = format_batch(
            requests,
            cfg=self.model.cfg,
            tokenizer=self.model.tokenizer,
            max_len=self.model.max_len,
        )
        labels = torch.tensor([ex.answer_index for ex in examples], dtype=torch.long)
        return DecisionBatch(formatted=formatted, labels=labels)


class EvalCollator:
    def __init__(self, model: ModernBERTDecisionModel) -> None:
        self.model = model

    def __call__(self, examples: list[DecisionExample]) -> DecisionBatch:
        requests = [example_to_request(ex) for ex in examples]
        formatted = format_batch(
            requests,
            cfg=self.model.cfg,
            tokenizer=self.model.tokenizer,
            max_len=self.model.max_len,
        )
        labels = torch.tensor([ex.answer_index for ex in examples], dtype=torch.long)
        return DecisionBatch(formatted=formatted, labels=labels)
