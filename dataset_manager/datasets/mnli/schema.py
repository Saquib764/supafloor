"""MNLI — premise/hypothesis pairs with entailment, neutral, or contradiction."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

from datasets import Dataset, Features, Sequence, Value, load_dataset

HUB_CONFIG = "mnli"
HUB_SPLITS = ("train", "validation_matched", "validation_mismatched")

QUESTION = "What is the relationship between the premise and hypothesis?"

OPTIONS = (
    "Premise implies the hypothesis",
    "The premise has insufficient information about hypothesis",
    "The hypothesis contradicts the premise.",
)

FEATURES = Features(
    {
        "example_id": Value("string"),
        "state": Value("string"),
        "question": Value("string"),
        "options": Sequence(Value("string")),
        "answer_index": Value("int64"),
    }
)


def split_jsonl_name(split: str) -> str:
    return f"{split}.jsonl"


def _format_state(premise: str, hypothesis: str) -> str:
    p = premise.strip()
    h = hypothesis.strip()
    if not p or not h:
        raise ValueError("MNLI premise and hypothesis must be non-blank")
    return f"Premise:\n{p}\nHypothesis:\n{h}"


def _row_from_hf(example: dict, *, example_id: str = "") -> dict:
    label = example.get("label")
    if label is None or int(label) < 0:
        raise ValueError("MNLI row missing gold label")
    answer_index = int(label)
    if not 0 <= answer_index < len(OPTIONS):
        raise ValueError(f"MNLI label out of range: {answer_index}")
    return {
        "example_id": example_id or str(example.get("idx") or ""),
        "state": _format_state(str(example["premise"]), str(example["hypothesis"])),
        "question": QUESTION,
        "options": list(OPTIONS),
        "answer_index": answer_index,
    }


def row_to_jsonl_line(row: dict) -> str:
    payload = {
        "example_id": str(row["example_id"]),
        "state": str(row["state"]),
        "question": str(row["question"]),
        "options": [str(o) for o in row["options"]],
        "answer_index": int(row["answer_index"]),
    }
    return json.dumps(payload, ensure_ascii=False)


def _jsonl_path(local_root: Path, split: str) -> Path:
    return Path(local_root) / split_jsonl_name(split)


def has_local_jsonl(local_root: Path, split: str) -> bool:
    return _jsonl_path(local_root, split).is_file()


def local_available(local_root: Path) -> bool:
    root = Path(local_root)
    if not root.is_dir():
        return False
    return any(has_local_jsonl(root, split) for split in HUB_SPLITS)


def jsonl_line_offsets(path: Path) -> list[int]:
    offsets: list[int] = []
    with path.open("rb") as handle:
        while True:
            pos = handle.tell()
            line = handle.readline()
            if not line:
                break
            if line.strip():
                offsets.append(pos)
    return offsets


def count_split_rows(local_root: Path, split: str) -> int:
    path = _jsonl_path(local_root, split)
    if not path.is_file():
        return 0
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def iter_split_rows(local_root: Path, split: str) -> Iterator[dict]:
    path = _jsonl_path(Path(local_root), split)
    if not path.is_file():
        raise FileNotFoundError(f"missing local split file: {path}")
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def iter_split_rows_shuffled(
    local_root: Path,
    split: str,
    *,
    seed: int,
) -> Iterator[dict]:
    import random

    path = _jsonl_path(Path(local_root), split)
    if not path.is_file():
        raise FileNotFoundError(f"missing local split file: {path}")
    offsets = jsonl_line_offsets(path)
    rng = random.Random(seed)
    rng.shuffle(offsets)
    with path.open("rb") as handle:
        for pos in offsets:
            handle.seek(pos)
            line = handle.readline().decode("utf-8").strip()
            if line:
                yield json.loads(line)


def _local_cache_version(src: Path) -> str:
    newest = 0
    for path in src.glob("*.jsonl"):
        newest = max(newest, path.stat().st_mtime_ns)
    return str(newest)


def build_from_local(local_root: Path) -> Dataset:
    local_root = Path(local_root)
    if not has_local_jsonl(local_root, "train"):
        raise FileNotFoundError(f"No MNLI train.jsonl under {local_root}")

    def _gen(src: Path, cache_version: str = ""):
        del cache_version
        yield from iter_split_rows(src, "train")

    dataset = Dataset.from_generator(
        _gen,
        gen_kwargs={"src": local_root, "cache_version": _local_cache_version(local_root)},
        features=FEATURES,
    )
    if len(dataset) == 0:
        raise FileNotFoundError(f"No MNLI rows under {local_root}")
    return dataset


def load_from_hub(
    repo: str,
    *,
    revision: str | None = "main",
    cache_dir: Path | None = None,
    split: str = "train",
    hub_config: str = HUB_CONFIG,
) -> Dataset:
    kwargs: dict[str, object] = {"split": split}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    if revision and str(revision).lower() not in ("null", "none", ""):
        kwargs["revision"] = revision
    token = os.environ.get("HF_TOKEN")
    if token:
        kwargs["token"] = token
    raw = load_dataset(repo, hub_config, **kwargs)
    rows = [
        _row_from_hf(dict(raw[int(i)]), example_id=f"{split}:{i}")
        for i in range(len(raw))
    ]
    return Dataset.from_list(rows, features=FEATURES)


def row_from_hf(example: dict) -> dict:
    return _row_from_hf(example)


def materialize_local(
    target: Path,
    *,
    repo: str,
    cache_dir: Path | None = None,
    revision: str | None = None,
    hub_config: str = HUB_CONFIG,
) -> int:
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    total = 0
    for split in HUB_SPLITS:
        dataset = load_from_hub(
            repo,
            revision=revision,
            cache_dir=cache_dir,
            split=split,
            hub_config=hub_config,
        )
        out_path = _jsonl_path(target, split)
        with out_path.open("w", encoding="utf-8") as handle:
            for row in dataset:
                handle.write(row_to_jsonl_line(row) + "\n")
        total += len(dataset)
        print(f"  {split}: {len(dataset):,} rows → {out_path.name}", flush=True)
    return total
