"""RACE reading comprehension — passage, question, four options, gold index."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

from datasets import Dataset, Features, Sequence, Value, load_dataset

RACE_SUBSET = "all"
HUB_SPLITS = ("train", "validation", "test")


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


def _answer_to_index(answer: str) -> int:
    letter = str(answer).strip().upper()
    if len(letter) != 1 or letter not in "ABCD":
        raise ValueError(f"RACE answer must be A–D, got {answer!r}")
    return ord(letter) - ord("A")


def _row_from_hf(example: dict) -> dict:
    options = [str(o).strip() for o in example["options"]]
    if len(options) < 2:
        raise ValueError("RACE example must have at least two options")
    return {
        "example_id": str(example.get("example_id") or example.get("id") or ""),
        "state": str(example["article"]).strip(),
        "question": str(example["question"]).strip(),
        "options": options,
        "answer_index": _answer_to_index(example["answer"]),
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


def _iter_cmu_passage_file(path: Path) -> Iterator[dict]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return
    data = json.loads(raw)
    article = str(data.get("article") or "").strip()
    questions = data.get("questions") or []
    options_list = data.get("options") or []
    answers = data.get("answers") or []
    if not article or not questions:
        return
    stem = path.stem
    for index, (question, options, answer) in enumerate(
        zip(questions, options_list, answers, strict=False)
    ):
        opts = [str(o).strip() for o in options]
        if not question or not opts or any(not o for o in opts):
            continue
        yield {
            "example_id": f"{stem}:{index}",
            "state": article,
            "question": str(question).strip(),
            "options": opts,
            "answer_index": _answer_to_index(answer),
        }


def _iter_cmu_tree(src: Path) -> Iterator[dict]:
    for path in sorted(src.rglob("*.txt")):
        if path.name.startswith("."):
            continue
        yield from _iter_cmu_passage_file(path)


def _jsonl_path(local_root: Path, split: str) -> Path:
    return Path(local_root) / split_jsonl_name(split)


def has_local_jsonl(local_root: Path, split: str) -> bool:
    return _jsonl_path(local_root, split).is_file()


def local_available(local_root: Path) -> bool:
    root = Path(local_root)
    if not root.is_dir():
        return False
    if any(has_local_jsonl(root, split) for split in HUB_SPLITS):
        return True
    return any(root.rglob("*.txt"))


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
    if path.is_file():
        count = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    count += 1
        return count
    if split != "train":
        return 0
    return sum(1 for _ in _iter_cmu_tree(Path(local_root)))


def iter_split_rows(local_root: Path, split: str) -> Iterator[dict]:
    root = Path(local_root)
    path = _jsonl_path(root, split)
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
        return
    if split != "train":
        raise FileNotFoundError(f"missing local split file: {path}")
    yield from _iter_cmu_tree(root)


def iter_split_rows_shuffled(
    local_root: Path,
    split: str,
    *,
    seed: int,
) -> Iterator[dict]:
    import random

    path = _jsonl_path(local_root, split)
    if not path.is_file():
        if split != "train":
            raise FileNotFoundError(f"missing local split file: {path}")
        rows = list(_iter_cmu_tree(Path(local_root)))
        rng = random.Random(seed)
        rng.shuffle(rows)
        yield from rows
        return

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
    for path in list(src.glob("*.jsonl")) + list(src.rglob("*.txt")):
        stat = path.stat()
        newest = max(newest, stat.st_mtime_ns)
    return str(newest)


def build_from_local(local_root: Path) -> Dataset:
    local_root = Path(local_root)
    if has_local_jsonl(local_root, "train"):

        def _gen(src: Path, cache_version: str = ""):
            del cache_version
            yield from iter_split_rows(src, "train")

        dataset = Dataset.from_generator(
            _gen,
            gen_kwargs={"src": local_root, "cache_version": _local_cache_version(local_root)},
            features=FEATURES,
        )
    else:

        def _gen_cmu(src: Path, cache_version: str = ""):
            del cache_version
            yield from _iter_cmu_tree(src)

        dataset = Dataset.from_generator(
            _gen_cmu,
            gen_kwargs={"src": local_root, "cache_version": _local_cache_version(local_root)},
            features=FEATURES,
        )
    if len(dataset) == 0:
        raise FileNotFoundError(f"No RACE rows under {local_root}")
    return dataset


def load_from_hub(
    repo: str,
    *,
    revision: str | None = "main",
    cache_dir: Path | None = None,
    split: str = "train",
    subset: str = RACE_SUBSET,
) -> Dataset:
    kwargs: dict[str, object] = {"split": split}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    if revision and str(revision).lower() not in ("null", "none", ""):
        kwargs["revision"] = revision
    token = os.environ.get("HF_TOKEN")
    if token:
        kwargs["token"] = token
    raw = load_dataset(repo, subset, **kwargs)
    rows = [_row_from_hf(raw[int(i)]) for i in range(len(raw))]
    return Dataset.from_list(rows, features=FEATURES)


def row_from_hf(example: dict) -> dict:
    """Map a raw Hub RACE row to registry FEATURES."""
    return _row_from_hf(example)


def materialize_local(
    target: Path,
    *,
    repo: str,
    subset: str = RACE_SUBSET,
    cache_dir: Path | None = None,
    revision: str | None = None,
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
            subset=subset,
        )
        out_path = _jsonl_path(target, split)
        with out_path.open("w", encoding="utf-8") as handle:
            for row in dataset:
                handle.write(row_to_jsonl_line(row) + "\n")
        total += len(dataset)
        print(f"  {split}: {len(dataset):,} rows → {out_path.name}", flush=True)
    return total


def write_local_folders(dataset: Dataset, target: Path) -> int:
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    out_path = _jsonl_path(target, "train")
    with out_path.open("w", encoding="utf-8") as handle:
        for row in dataset:
            handle.write(row_to_jsonl_line(row) + "\n")
    return len(dataset)
