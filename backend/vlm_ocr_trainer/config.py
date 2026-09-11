"""Load vlm_ocr_trainer/config.yaml into typed settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

_PACKAGE_DIR = Path(__file__).resolve().parent
_STEM_STRIDE = 16


def _project_dir() -> Path:
    if (_PACKAGE_DIR / "pyproject.toml").is_file():
        return _PACKAGE_DIR
    cwd = Path.cwd()
    if (cwd / "config.yaml").is_file():
        return cwd
    return _PACKAGE_DIR


CONFIG_PATH = _project_dir() / "config.yaml"


@dataclass(frozen=True)
class DataSettings:
    seed: int | None
    image_sizes: tuple[tuple[int, int], ...]
    sentence_len_min: int
    sentence_len_max: int
    min_lines: int
    max_lines: int
    samples_per_epoch: int


@dataclass(frozen=True)
class TrainSettings:
    session_dir: Path
    epochs: int
    batch_size: int
    grad_accum: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    bf16: bool
    resume: bool
    save_steps: int
    viz_steps: int
    viz_samples: int
    max_grad_norm: float
    num_workers: int


@dataclass(frozen=True)
class InferSettings:
    checkpoint: Path


@dataclass(frozen=True)
class Config:
    max_seq_len: int
    data: DataSettings
    train: TrainSettings
    infer: InferSettings

    @property
    def max_image_width(self) -> int:
        return max(width for width, _height in self.data.image_sizes)

    @property
    def max_image_height(self) -> int:
        return max(height for _width, height in self.data.image_sizes)


def _resolve(path: Path, base: Path) -> Path:
    return path if path.is_absolute() else (base / path).resolve()


def _parse_image_sizes(raw: Any) -> tuple[tuple[int, int], ...]:
    if not raw:
        raise ValueError("data.image_sizes must not be empty")
    sizes: list[tuple[int, int]] = []
    for entry in raw:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError(f"each data.image_sizes entry must be [width, height], got {entry!r}")
        width, height = int(entry[0]), int(entry[1])
        if width < _STEM_STRIDE or height < _STEM_STRIDE:
            raise ValueError(f"image size {width}x{height} must be at least {_STEM_STRIDE}x{_STEM_STRIDE}")
        if width % _STEM_STRIDE or height % _STEM_STRIDE:
            raise ValueError(f"image size {width}x{height} must be divisible by {_STEM_STRIDE}")
        sizes.append((width, height))
    return tuple(sizes)


def load_config(path: Path | None = None) -> Config:
    config_path = path or CONFIG_PATH
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    base = config_path.parent
    data = raw["data"]
    train = raw["train"]
    infer = raw.get("infer") or {}
    min_lines = int(data["min_lines"])
    max_lines = int(data["max_lines"])
    if min_lines < 1:
        raise ValueError("data.min_lines must be >= 1")
    if max_lines < min_lines:
        raise ValueError("data.max_lines must be >= data.min_lines")
    sentence_len_min = int(data["sentence_len_min"])
    sentence_len_max = int(data["sentence_len_max"])
    if sentence_len_min < 1:
        raise ValueError("data.sentence_len_min must be >= 1")
    if sentence_len_max < sentence_len_min:
        raise ValueError("data.sentence_len_max must be >= data.sentence_len_min")
    return Config(
        max_seq_len=int(raw["max_seq_len"]),
        data=DataSettings(
            seed=int(data["seed"]) if data.get("seed") is not None else None,
            image_sizes=_parse_image_sizes(data.get("image_sizes") or raw.get("image_sizes")),
            sentence_len_min=sentence_len_min,
            sentence_len_max=sentence_len_max,
            min_lines=min_lines,
            max_lines=max_lines,
            samples_per_epoch=int(data["samples_per_epoch"]),
        ),
        train=TrainSettings(
            session_dir=_resolve(Path(train["session_dir"]), base),
            epochs=int(train["epochs"]),
            batch_size=int(train["batch_size"]),
            grad_accum=max(1, int(train.get("grad_accum", 1))),
            learning_rate=float(train["learning_rate"]),
            weight_decay=float(train["weight_decay"]),
            warmup_ratio=float(train.get("warmup_ratio", train.get("warmup_fraction", 0.03))),
            bf16=bool(train["bf16"]),
            resume=bool(train.get("resume", False)),
            save_steps=int(train["save_steps"]),
            viz_steps=int(train.get("viz_steps", train.get("sample_steps", 200))),
            viz_samples=int(train.get("viz_samples", 4)),
            max_grad_norm=float(train["max_grad_norm"]),
            num_workers=int(train.get("num_workers", 0)),
        ),
        infer=InferSettings(
            checkpoint=_resolve(Path(infer.get("checkpoint") or "training_session/last.pt"), base),
        ),
    )
