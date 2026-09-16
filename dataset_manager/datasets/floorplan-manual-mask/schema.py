"""Manual mask Hub schema: canvas image + five PNG mask layers."""

from __future__ import annotations

from pathlib import Path

from datasets import Dataset, Features, Image, Value, load_dataset

JPEG_QUALITY = 95
IMAGE_EXT = ".jpg"
MASK_NAMES = ("background", "room", "wall", "door", "window")

FEATURES = Features(
    {
        "stem": Value("string"),
        "image": Image(),
        "background": Image(),
        "room": Image(),
        "wall": Image(),
        "door": Image(),
        "window": Image(),
    }
)


def _iter_local_rows(src: Path):
    output_dir = src / "output"
    for image_path in sorted((src / "input").glob(f"*{IMAGE_EXT}")):
        stem = image_path.stem
        mask_dir = output_dir / stem
        mask_paths = {name: mask_dir / f"{name}.png" for name in MASK_NAMES}
        if not mask_dir.is_dir() or not all(path.is_file() for path in mask_paths.values()):
            continue
        yield {
            "stem": stem,
            "image": str(image_path),
            **{name: str(mask_paths[name]) for name in MASK_NAMES},
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
        row["image"].convert("RGB").save(
            input_dir / f"{stem}{IMAGE_EXT}", format="JPEG", quality=JPEG_QUALITY
        )
        mask_dir = output_dir / stem
        mask_dir.mkdir(parents=True, exist_ok=True)
        for name in MASK_NAMES:
            row[name].convert("L").save(mask_dir / f"{name}.png")
        written += 1
    return written
