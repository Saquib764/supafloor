"""Load vlm_ocr_trainer/config.yaml into typed settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

import yaml

from model import DECODER_MAX_SEQ_LEN, STEM_STRIDE_H, STEM_STRIDE_W, validate_image_size

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

_PACKAGE_DIR = Path(__file__).resolve().parent

T = TypeVar("T")


def _project_dir() -> Path:
    if (_PACKAGE_DIR / "pyproject.toml").is_file():
        return _PACKAGE_DIR
    cwd = Path.cwd()
    if (cwd / "config.yaml").is_file():
        return cwd
    return _PACKAGE_DIR


CONFIG_PATH = _project_dir() / "config.yaml"


@dataclass(frozen=True)
class StepSchedule(Generic[T]):
    """Piecewise-constant schedule: value at training step >= threshold."""

    thresholds: tuple[tuple[int, T], ...]

    def at(self, step: int) -> T:
        value = self.thresholds[0][1]
        for threshold, scheduled in self.thresholds:
            if step >= threshold:
                value = scheduled
            else:
                break
        return value

    @property
    def final(self) -> T:
        return self.thresholds[-1][1]


@dataclass(frozen=True)
class DataAtStep:
    image_sizes: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class DataSettings:
    seed: int | None
    image_sizes: StepSchedule[tuple[tuple[int, int], ...]]
    paragraph_counts: tuple[int, ...]
    paragraph_len_min: int
    paragraph_len_max: int
    samples_per_epoch: int

    def at_step(self, step: int) -> DataAtStep:
        return DataAtStep(image_sizes=self.image_sizes.at(step))

    @property
    def final_step(self) -> int:
        return self.image_sizes.thresholds[-1][0]

    def at_final(self) -> DataAtStep:
        return self.at_step(self.final_step)

    @property
    def all_image_sizes(self) -> tuple[tuple[int, int], ...]:
        seen: set[tuple[int, int]] = set()
        ordered: list[tuple[int, int]] = []
        for _step, sizes in self.image_sizes.thresholds:
            for size in sizes:
                if size not in seen:
                    seen.add(size)
                    ordered.append(size)
        return tuple(ordered)


@dataclass(frozen=True)
class WordsSettings:
    path: Path
    num_topics: int
    words_per_topic: int
    seed: int


@dataclass(frozen=True)
class HoldoutSettings:
    dir: Path
    seed: int
    num_samples: int
    width: int
    height: int
    paragraph_counts: tuple[int, ...]
    paragraph_len_min: int
    paragraph_len_max: int
    words: Path


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
    prefetch_factor: int
    prefetch_samples: int
    benchmark_steps: int
    benchmark_batch_size: int | None
    save_best: bool


@dataclass(frozen=True)
class InferSettings:
    checkpoint: Path


@dataclass(frozen=True)
class Config:
    max_seq_len: StepSchedule[int]
    data: DataSettings
    words: WordsSettings
    holdout: HoldoutSettings
    train: TrainSettings
    infer: InferSettings

    @property
    def max_image_width(self) -> int:
        return max(width for width, _height in self.data.all_image_sizes)

    @property
    def max_image_height(self) -> int:
        return max(height for _width, height in self.data.all_image_sizes)


def _resolve(path: Path, base: Path) -> Path:
    return path if path.is_absolute() else (base / path).resolve()


def _is_step_schedule(raw: Any) -> bool:
    if not isinstance(raw, list) or not raw:
        return False
    return all(
        isinstance(entry, (list, tuple))
        and len(entry) == 2
        and isinstance(entry[0], int)
        for entry in raw
    )


def _validate_step_schedule(
    thresholds: list[tuple[int, Any]],
    field_name: str,
) -> tuple[tuple[int, Any], ...]:
    if not thresholds:
        raise ValueError(f"{field_name} step schedule must not be empty")
    if thresholds[0][0] != 0:
        raise ValueError(f"{field_name} step schedule must start at step 0")
    for index in range(1, len(thresholds)):
        if thresholds[index][0] <= thresholds[index - 1][0]:
            raise ValueError(f"{field_name} step thresholds must be strictly increasing")
    return tuple(thresholds)


def _parse_step_schedule(
    raw: Any,
    *,
    field_name: str,
    parse_value: Any,
) -> StepSchedule[Any]:
    if _is_step_schedule(raw):
        thresholds = _validate_step_schedule(
            [(int(entry[0]), parse_value(entry[1])) for entry in raw],
            field_name,
        )
        return StepSchedule(thresholds)
    return StepSchedule(((0, parse_value(raw)),))


def _parse_image_sizes_value(raw: Any) -> tuple[tuple[int, int], ...]:
    if not raw:
        raise ValueError("data.image_sizes must not be empty")
    sizes: list[tuple[int, int]] = []
    for entry in raw:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError(f"each data.image_sizes entry must be [width, height], got {entry!r}")
        width, height = int(entry[0]), int(entry[1])
        validate_image_size(width, height)
        sizes.append((width, height))
    return tuple(sizes)


def _is_image_size_schedule(raw: Any) -> bool:
    if not _is_step_schedule(raw):
        return False
    _step, value = raw[0]
    return isinstance(value, list)


def _parse_int_value(raw: Any, *, field_name: str) -> int:
    value = int(raw)
    if value < 4:
        raise ValueError(f"{field_name} must be >= 4")
    return value


def _parse_int_schedule(raw: Any, *, field_name: str) -> StepSchedule[int]:
    if _is_step_schedule(raw):
        return _parse_step_schedule(
            raw,
            field_name=field_name,
            parse_value=lambda value: _parse_int_value(value, field_name=field_name),
        )
    return StepSchedule(((0, _parse_int_value(raw, field_name=field_name)),))


def _parse_image_sizes(raw: Any) -> StepSchedule[tuple[tuple[int, int], ...]]:
    if _is_image_size_schedule(raw):
        return _parse_step_schedule(
            raw,
            field_name="data.image_sizes",
            parse_value=_parse_image_sizes_value,
        )
    return StepSchedule(((0, _parse_image_sizes_value(raw)),))


def _parse_counts_value(raw: Any, *, field_name: str) -> tuple[int, ...]:
    counts = tuple(int(value) for value in raw)
    if not counts:
        raise ValueError(f"{field_name} must not be empty")
    if any(value < 1 for value in counts):
        raise ValueError(f"each {field_name} entry must be >= 1")
    return counts


def _parse_paragraph_render(raw: dict[str, Any], *, field_prefix: str) -> dict[str, Any]:
    paragraph_len_min = int(raw.get("paragraph_len_min", 120))
    paragraph_len_max = int(raw.get("paragraph_len_max", 400))
    if paragraph_len_min < 1:
        raise ValueError(f"{field_prefix}.paragraph_len_min must be >= 1")
    if paragraph_len_max < paragraph_len_min:
        raise ValueError(f"{field_prefix}.paragraph_len_max must be >= {field_prefix}.paragraph_len_min")
    paragraph_counts = _parse_counts_value(
        raw.get("paragraph_counts", [1, 2, 3]),
        field_name=f"{field_prefix}.paragraph_counts",
    )
    return {
        "paragraph_counts": paragraph_counts,
        "paragraph_len_min": paragraph_len_min,
        "paragraph_len_max": paragraph_len_max,
    }


def _parse_single_image_size(raw: Any, *, field_name: str) -> tuple[int, int]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(f"{field_name} must be [width, height]")
    return _parse_image_sizes_value([raw])[0]


def _parse_holdout(holdout: dict[str, Any], *, words_path: Path) -> dict[str, Any]:
    image_size = holdout.get("image_size")
    if image_size is None:
        width = int(holdout["width"])
        height = int(holdout["height"])
        validate_image_size(width, height)
    else:
        width, height = _parse_single_image_size(image_size, field_name="holdout.image_size")
    render = _parse_paragraph_render(holdout, field_prefix="holdout")
    return {
        "width": width,
        "height": height,
        "words": words_path,
        **render,
    }


def _format_step_schedule(name: str, schedule: StepSchedule[Any]) -> str:
    parts = [f"step>={threshold}: {value!r}" for threshold, value in schedule.thresholds]
    return f"{name + ':':<16} {' | '.join(parts)}"


def print_data_schedule(cfg: Config) -> None:
    data = cfg.data
    print(
        f"stem_stride:      W÷{STEM_STRIDE_W}, H÷{STEM_STRIDE_H}  "
        f"(256×256 → 64×32 visual grid)",
    )
    print(_format_step_schedule("image_sizes", data.image_sizes))
    print(f"paragraph_counts: {list(data.paragraph_counts)}")
    print(f"paragraph_len:    {data.paragraph_len_min}–{data.paragraph_len_max}")
    print("font_size:        8–16 px (random per sample, bold/italic/handwritten)")
    print("variations:       letter/line spacing, AA, blur, noise, JPEG, rotation, resize")
    print(_format_step_schedule("max_seq_len", cfg.max_seq_len))


def load_config(path: Path | None = None) -> Config:
    config_path = path or CONFIG_PATH
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    base = config_path.parent
    data = raw["data"]
    holdout = raw.get("holdout") or {}
    words = raw.get("words") or {}
    train = raw["train"]
    infer = raw.get("infer") or {}
    data_render = _parse_paragraph_render(data, field_prefix="data")
    words_path = _resolve(Path(words.get("path") or holdout.get("words") or "renderer/words.json"), base)
    num_topics = int(words.get("num_topics", 24))
    if num_topics < 1:
        raise ValueError("words.num_topics must be >= 1")
    words_per_topic = int(words.get("words_per_topic", 100))
    if words_per_topic < 10:
        raise ValueError("words.words_per_topic must be >= 10")
    holdout_render = _parse_holdout(holdout, words_path=words_path)
    holdout_num_samples = int(holdout.get("num_samples", 512))
    if holdout_num_samples < 1:
        raise ValueError("holdout.num_samples must be >= 1")
    benchmark_steps = int(train.get("benchmark_steps", 0))
    if benchmark_steps < 0:
        raise ValueError("train.benchmark_steps must be >= 0")
    benchmark_batch_size_raw = train.get("benchmark_batch_size")
    benchmark_batch_size = (
        int(benchmark_batch_size_raw) if benchmark_batch_size_raw is not None else None
    )
    if benchmark_batch_size is not None and benchmark_batch_size < 1:
        raise ValueError("train.benchmark_batch_size must be >= 1")
    num_workers = int(train.get("num_workers", 0))
    if num_workers < 0:
        raise ValueError("train.num_workers must be >= 0")
    prefetch_factor = int(train.get("prefetch_factor", 2))
    if prefetch_factor < 1:
        raise ValueError("train.prefetch_factor must be >= 1")
    prefetch_samples = int(train.get("prefetch_samples", 0))
    if prefetch_samples < 0:
        raise ValueError("train.prefetch_samples must be >= 0")
    max_seq_len = _parse_int_schedule(raw["max_seq_len"], field_name="max_seq_len")
    if max_seq_len.final > DECODER_MAX_SEQ_LEN:
        raise ValueError(
            f"max_seq_len schedule final ({max_seq_len.final}) exceeds "
            f"DECODER_MAX_SEQ_LEN ({DECODER_MAX_SEQ_LEN}) in model.py",
        )
    return Config(
        max_seq_len=max_seq_len,
        data=DataSettings(
            seed=int(data["seed"]) if data.get("seed") is not None else None,
            image_sizes=_parse_image_sizes(data.get("image_sizes") or raw.get("image_sizes")),
            samples_per_epoch=int(data["samples_per_epoch"]),
            **data_render,
        ),
        words=WordsSettings(
            path=words_path,
            num_topics=num_topics,
            words_per_topic=words_per_topic,
            seed=int(words.get("seed", 42)),
        ),
        holdout=HoldoutSettings(
            dir=_resolve(Path(holdout.get("dir") or "holdout"), base),
            seed=int(holdout.get("seed", 900_001)),
            num_samples=holdout_num_samples,
            **holdout_render,
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
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            prefetch_samples=prefetch_samples,
            benchmark_steps=benchmark_steps,
            benchmark_batch_size=benchmark_batch_size,
            save_best=bool(train.get("save_best", True)),
        ),
        infer=InferSettings(
            checkpoint=_resolve(Path(infer.get("checkpoint") or "training_session/last.pt"), base),
        ),
    )
