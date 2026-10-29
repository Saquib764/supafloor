"""Load train/test CSV; batch sampled authors on sampled prompts."""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import PreTrainedTokenizerBase

from config import Config
from labels import NUM_CLASSES, model_to_index

MODELS_PER_PROMPT = NUM_CLASSES


@dataclass(frozen=True)
class PassageExample:
    row_id: str
    prompt_id: str
    text: str
    label_index: int


@dataclass
class ClassifierBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


def read_passage_csv(path: Path) -> list[PassageExample]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    out: list[PassageExample] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out.append(
                PassageExample(
                    row_id=str(row["id"]),
                    prompt_id=str(row["prompt_id"]).strip(),
                    text=str(row["text"]).strip(),
                    label_index=model_to_index(str(row["model"]).strip()),
                )
            )
    if not out:
        raise ValueError(f"no rows in {path}")
    return out


def read_train_csv(cfg: Config) -> list[PassageExample]:
    return read_passage_csv(cfg.data.root / cfg.data.train_csv)


def read_test_csv(cfg: Config) -> list[PassageExample]:
    return read_passage_csv(cfg.data.root / cfg.data.test_csv)


def _prompt_is_complete(rows: list[PassageExample]) -> bool:
    if len(rows) != MODELS_PER_PROMPT:
        return False
    return len({ex.label_index for ex in rows}) == MODELS_PER_PROMPT


def filter_complete_prompts(examples: list[PassageExample]) -> list[PassageExample]:
    by_prompt: dict[str, list[PassageExample]] = defaultdict(list)
    for ex in examples:
        by_prompt[ex.prompt_id].append(ex)
    out: list[PassageExample] = []
    for rows in by_prompt.values():
        if _prompt_is_complete(rows):
            out.extend(rows)
    return out


def split_train_test(
    examples: list[PassageExample],
    *,
    test_fraction: float,
    seed: int,
) -> tuple[list[PassageExample], list[PassageExample]]:
    complete = filter_complete_prompts(examples)
    if not complete:
        raise ValueError("no complete prompts (need one row per model per prompt_id)")
    by_prompt: dict[str, list[PassageExample]] = defaultdict(list)
    for ex in complete:
        by_prompt[ex.prompt_id].append(ex)
    prompt_ids = sorted(by_prompt.keys())
    if test_fraction <= 0.0:
        if not prompt_ids:
            raise ValueError("need at least one complete prompt")
        train = [ex for pid in prompt_ids for ex in by_prompt[pid]]
        return train, []
    if len(prompt_ids) < 2:
        raise ValueError("need at least two complete prompts for train/test split")
    rng = random.Random(seed)
    rng.shuffle(prompt_ids)
    n_test = max(1, int(len(prompt_ids) * test_fraction))
    if len(prompt_ids) - n_test < 1:
        raise ValueError("train split needs at least one complete prompt")
    test_pids = set(prompt_ids[:n_test])
    train: list[PassageExample] = []
    test: list[PassageExample] = []
    for pid in prompt_ids:
        rows = by_prompt[pid]
        if pid in test_pids:
            test.extend(rows)
        else:
            train.extend(rows)
    return train, test


def load_train_test_split(cfg: Config) -> tuple[list[PassageExample], list[PassageExample]]:
    train = filter_complete_prompts(read_train_csv(cfg))
    if not train:
        raise ValueError(
            f"no complete prompts in {cfg.data.root / cfg.data.train_csv} "
            f"(need {MODELS_PER_PROMPT} distinct models per prompt_id)"
        )
    if cfg.data.test_fraction <= 0.0:
        return train, []
    _, test = split_train_test(
        read_test_csv(cfg),
        test_fraction=cfg.data.test_fraction,
        seed=cfg.data.seed,
    )
    return train, test


def build_prompt_label_index(examples: list[PassageExample]) -> dict[str, dict[int, int]]:
    """prompt_id -> label_index -> row index in `examples`."""
    index: dict[str, dict[int, int]] = {}
    for i, ex in enumerate(examples):
        slot = index.setdefault(ex.prompt_id, {})
        label = int(ex.label_index)
        if label in slot:
            raise ValueError(f"duplicate model {label} for prompt {ex.prompt_id}")
        slot[label] = i
    for pid, labels in index.items():
        if len(labels) != MODELS_PER_PROMPT:
            raise ValueError(
                f"prompt {pid} has {len(labels)} models; expected {MODELS_PER_PROMPT}"
            )
    return index


def iter_prompt_batches(
    examples: list[PassageExample],
    *,
    batch_size: int,
    prompt_count: int,
    seed: int,
    epoch: int = 0,
) -> Iterator[list[int]]:
    """Sample authors_per_batch models and prompt_count prompts; one row per pair.

    authors_per_batch = batch_size / prompt_count. Only (prompt, author) cells for
    the sampled sets are included, not every model on those prompts.
    """
    if prompt_count < 1:
        raise ValueError("prompt_count must be >= 1")
    if batch_size < 1 or batch_size % prompt_count != 0:
        raise ValueError("batch_size must be a positive multiple of prompt_count")
    authors_per_batch = batch_size // prompt_count
    if authors_per_batch > MODELS_PER_PROMPT:
        raise ValueError(
            f"batch_size / prompt_count is {authors_per_batch}; "
            f"each prompt has {MODELS_PER_PROMPT} models"
        )
    by_prompt = build_prompt_label_index(examples)
    prompt_ids = sorted(by_prompt.keys())
    if len(prompt_ids) < prompt_count:
        return
    rng = random.Random(seed + int(epoch) * 1_000_003)
    rng.shuffle(prompt_ids)
    n_groups = len(prompt_ids) // prompt_count
    prompt_ids = prompt_ids[: n_groups * prompt_count]
    all_models = list(range(MODELS_PER_PROMPT))
    for group_i in range(n_groups):
        group = prompt_ids[group_i * prompt_count : (group_i + 1) * prompt_count]
        authors = rng.sample(all_models, authors_per_batch)
        rows: list[int] = []
        for pid in group:
            for label in authors:
                rows.append(by_prompt[pid][label])
        yield rows


def count_prompt_batches(
    examples: list[PassageExample], *, batch_size: int, prompt_count: int
) -> int:
    if batch_size % prompt_count != 0:
        raise ValueError("batch_size must be a positive multiple of prompt_count")
    n_prompts = len(build_prompt_label_index(examples))
    return n_prompts // prompt_count


class PromptBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Random author and prompt samples; batch_size = prompt_count × authors_per_batch."""

    def __init__(
        self,
        examples: list[PassageExample],
        *,
        batch_size: int,
        prompt_count: int,
        seed: int,
    ) -> None:
        if prompt_count < 1:
            raise ValueError("prompt_count must be >= 1")
        if batch_size < 1 or batch_size % prompt_count != 0:
            raise ValueError("batch_size must be a positive multiple of prompt_count")
        self._examples = examples
        self._batch_size = batch_size
        self._prompt_count = prompt_count
        self._seed = seed
        self._authors_per_batch = batch_size // prompt_count
        self._num_batches = count_prompt_batches(
            examples, batch_size=batch_size, prompt_count=prompt_count
        )
        if self._num_batches < 1:
            raise ValueError(
                f"need at least {prompt_count} complete prompts; "
                f"found {len(examples) // MODELS_PER_PROMPT}"
            )
        self._batches: list[list[int]] = []
        self.set_epoch(0)

    def __len__(self) -> int:
        return self._num_batches

    def set_epoch(self, epoch: int) -> None:
        self._batches = list(
            iter_prompt_batches(
                self._examples,
                batch_size=self._batch_size,
                prompt_count=self._prompt_count,
                seed=self._seed,
                epoch=int(epoch),
            )
        )

    def __iter__(self):
        yield from self._batches


class PassageDataset(torch.utils.data.Dataset):
    def __init__(self, examples: list[PassageExample]) -> None:
        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> PassageExample:
        return self._examples[int(index)]


def prepare_tokenizer(tokenizer: PreTrainedTokenizerBase, *, max_len: int) -> PreTrainedTokenizerBase:
    if tokenizer.pad_token_id is None:
        eos_id = tokenizer.eos_token_id
        if eos_id is None:
            raise ValueError("tokenizer needs pad_token_id or eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = max_len
    tokenizer.padding_side = "right"
    return tokenizer


class PassageCollator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_len: int,
        probe_slots: int = 0,
    ) -> None:
        self.tokenizer = tokenizer
        if probe_slots < 0:
            raise ValueError("probe_slots must be >= 0")
        self.max_len = max_len - probe_slots

    def __call__(self, examples: list[PassageExample]) -> ClassifierBatch:
        texts = []
        for ex in examples:
            body = ex.text.strip()
            if not body:
                raise ValueError(f"row {ex.row_id} has empty text")
            texts.append(body)
        encoded = self.tokenizer(
            texts,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            padding=True,
            return_tensors="pt",
        )
        labels = torch.tensor([ex.label_index for ex in examples], dtype=torch.long)
        return ClassifierBatch(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            labels=labels,
        )
