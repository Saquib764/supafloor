"""LLM decision corpus: context paragraph + yes/no question."""

from __future__ import annotations

import json
import os
from pathlib import Path

from datasets import Dataset, Features, Value, load_dataset

SENTENCES_FILE = "sentences.jsonl"

FEATURES = Features(
    {
        "context": Value("string"),
        "question": Value("string"),
        "output": Value("int64"),
    }
)


def local_available(local_root: Path) -> bool:
    return (Path(local_root) / SENTENCES_FILE).is_file()


def _local_cache_version(src: Path) -> str:
    path = src / SENTENCES_FILE
    stat = path.stat()
    return f"{stat.st_mtime_ns}-{stat.st_size}"


def _iter_local_rows(src: Path, *, cache_version: str = ""):
    del cache_version  # busts HuggingFace generator cache when sentences.jsonl changes
    path = src / SENTENCES_FILE
    if not path.is_file():
        raise FileNotFoundError(f"Missing {SENTENCES_FILE} under {src}")
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        context = str(row.get("context", "")).strip()
        question = str(row.get("question", "")).strip()
        if not context or not question:
            continue
        output = int(row["output"])
        if output not in (0, 1):
            raise ValueError(f"output must be 0 or 1, got {output!r}")
        yield {"context": context, "question": question, "output": output}


def build_from_local(local_root: Path) -> Dataset:
    local_root = Path(local_root)
    dataset = Dataset.from_generator(
        _iter_local_rows,
        gen_kwargs={
            "src": local_root,
            "cache_version": _local_cache_version(local_root),
        },
        features=FEATURES,
    )
    if len(dataset) == 0:
        raise FileNotFoundError(f"No rows in {local_root / SENTENCES_FILE}")
    return dataset


def load_from_hub(repo: str, *, revision: str = "main", cache_dir: Path | None = None) -> Dataset:
    kwargs: dict[str, object] = {"split": "train", "revision": revision}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    token = os.environ.get("HF_TOKEN")
    if token:
        kwargs["token"] = token
    dataset = load_dataset(repo, **kwargs)
    columns = list(FEATURES.keys())
    missing = [col for col in columns if col not in dataset.column_names]
    if missing:
        raise KeyError(f"Hub dataset missing columns: {missing}")
    return dataset.select_columns(columns)


def write_local_folders(dataset: Dataset, target: Path) -> int:
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    path = target / SENTENCES_FILE
    lines = [
        json.dumps(
            {
                "context": str(row["context"]),
                "question": str(row["question"]),
                "output": int(row["output"]),
            },
            ensure_ascii=False,
        )
        for row in dataset
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)
