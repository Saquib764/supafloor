"""Floorplan-to-json Hub schema: image + extraction JSON."""

from __future__ import annotations

from pathlib import Path

from datasets import Dataset, Features, Image, Value, load_dataset

FEATURES = Features(
    {
        "stem": Value("string"),
        "plan_id": Value("string"),
        "variant": Value("int32"),
        "image": Image(),
        "extraction": Value("string"),
    }
)


def split_stem(stem: str) -> tuple[str, int]:
    if "_" not in stem:
        return stem, 0
    plan_id, rest = stem.rsplit("_", 1)
    try:
        return plan_id, int(rest)
    except ValueError:
        return stem, 0


def _iter_local_rows(src: Path):
    output_dir = src / "output"
    for png in sorted((src / "input").glob("*.png")):
        stem = png.stem
        json_path = output_dir / f"{stem}.json"
        if not json_path.is_file():
            continue
        plan_id, variant = split_stem(stem)
        yield {
            "stem": stem,
            "plan_id": plan_id,
            "variant": variant,
            "image": str(png),
            "extraction": json_path.read_text(encoding="utf-8"),
        }


def build_from_local(local_root: Path) -> Dataset:
    local_root = Path(local_root)
    if not (local_root / "input").is_dir():
        raise FileNotFoundError(f"Missing input/ under {local_root}")
    dataset = Dataset.from_generator(
        _iter_local_rows,
        gen_kwargs={"src": local_root},
        features=FEATURES,
    )
    if len(dataset) == 0:
        raise FileNotFoundError(f"No complete input/output pairs under {local_root}")
    return dataset


def load_from_hub(repo: str, *, revision: str = "main", cache_dir: Path | None = None) -> Dataset:
    kwargs: dict[str, object] = {"split": "train", "revision": revision}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    dataset = load_dataset(repo, **kwargs)
    columns = list(FEATURES.keys())
    missing = [col for col in columns if col not in dataset.column_names]
    if missing:
        raise KeyError(f"Hub dataset missing columns: {missing}")
    return dataset.select_columns(columns)


def write_local_folders(dataset: Dataset, target: Path) -> int:
    target = Path(target)
    input_dir = target / "input"
    output_dir = target / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for row in dataset:
        stem = str(row["stem"])
        row["image"].convert("RGB").save(input_dir / f"{stem}.png")
        (output_dir / f"{stem}.json").write_text(row["extraction"], encoding="utf-8")
        written += 1
    return written
