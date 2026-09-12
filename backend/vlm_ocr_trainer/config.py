"""Load vlm_ocr_trainer/config.yaml into typed settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

import yaml

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

_PACKAGE_DIR = Path(__file__).resolve().parent
_STEM_STRIDE = 16

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
class RenderVariations:
    """Per-effect on/off flags for render-time randomization."""

    background: bool = False
    grain: bool = False
    ink: bool = False
    handwritten: bool = False
    font: bool = False
    font_size: bool = False
    x_offset: bool = False
    line_spacing: bool = False


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
    sentence_len_min: int
    sentence_len_max: int
    line_counts: tuple[int, ...]
    variations: RenderVariations


@dataclass(frozen=True)
class VariationSchedules:
    background: StepSchedule[bool]
    grain: StepSchedule[bool]
    ink: StepSchedule[bool]
    handwritten: StepSchedule[bool]
    font: StepSchedule[bool]
    font_size: StepSchedule[bool]
    x_offset: StepSchedule[bool]
    line_spacing: StepSchedule[bool]

    def at(self, step: int) -> RenderVariations:
        return RenderVariations(
            background=self.background.at(step),
            grain=self.grain.at(step),
            ink=self.ink.at(step),
            handwritten=self.handwritten.at(step),
            font=self.font.at(step),
            font_size=self.font_size.at(step),
            x_offset=self.x_offset.at(step),
            line_spacing=self.line_spacing.at(step),
        )

    @property
    def final_step(self) -> int:
        return max(
            schedule.thresholds[-1][0]
            for schedule in (
                self.background,
                self.grain,
                self.ink,
                self.handwritten,
                self.font,
                self.font_size,
                self.x_offset,
                self.line_spacing,
            )
        )

    @property
    def final(self) -> RenderVariations:
        return self.at(self.final_step)


@dataclass(frozen=True)
class DataSettings:
    seed: int | None
    image_sizes: StepSchedule[tuple[tuple[int, int], ...]]
    sentence_len_min: StepSchedule[int]
    sentence_len_max: StepSchedule[int]
    line_counts: StepSchedule[tuple[int, ...]]
    variations: VariationSchedules
    samples_per_epoch: int

    def at_step(self, step: int) -> DataAtStep:
        sentence_len_min = self.sentence_len_min.at(step)
        sentence_len_max = self.sentence_len_max.at(step)
        if sentence_len_max < sentence_len_min:
            raise ValueError(
                f"at step {step}: sentence_len_max ({sentence_len_max}) "
                f"< sentence_len_min ({sentence_len_min})"
            )
        return DataAtStep(
            image_sizes=self.image_sizes.at(step),
            sentence_len_min=sentence_len_min,
            sentence_len_max=sentence_len_max,
            line_counts=self.line_counts.at(step),
            variations=self.variations.at(step),
        )

    @property
    def final_step(self) -> int:
        steps = [
            schedule.thresholds[-1][0]
            for schedule in (
                self.image_sizes,
                self.sentence_len_min,
                self.sentence_len_max,
                self.line_counts,
                self.variations.background,
                self.variations.grain,
                self.variations.ink,
                self.variations.handwritten,
                self.variations.font,
                self.variations.font_size,
                self.variations.x_offset,
                self.variations.line_spacing,
            )
        ]
        return max(steps)

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
class HoldoutSettings:
    dir: Path
    seed: int
    num_samples: int
    width: int
    height: int
    sentence_len_min: int
    sentence_len_max: int
    line_counts: tuple[int, ...]
    variations: RenderVariations


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
    max_seq_len: int
    data: DataSettings
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


def _parse_bool(raw: Any, *, field_name: str) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)) and raw in (0, 1):
        return bool(raw)
    raise ValueError(f"{field_name} must be a boolean")


def _parse_image_sizes_value(raw: Any) -> tuple[tuple[int, int], ...]:
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


def _is_image_size_schedule(raw: Any) -> bool:
    if not _is_step_schedule(raw):
        return False
    _step, value = raw[0]
    return isinstance(value, list)


def _parse_image_sizes(raw: Any) -> StepSchedule[tuple[tuple[int, int], ...]]:
    if _is_image_size_schedule(raw):
        return _parse_step_schedule(
            raw,
            field_name="data.image_sizes",
            parse_value=_parse_image_sizes_value,
        )
    return StepSchedule(((0, _parse_image_sizes_value(raw)),))


def _parse_line_counts_value(raw: Any) -> tuple[int, ...]:
    line_counts = tuple(int(value) for value in raw)
    if not line_counts:
        raise ValueError("data.line_counts must not be empty")
    if any(value < 1 for value in line_counts):
        raise ValueError("each data.line_counts entry must be >= 1")
    return line_counts


def _parse_line_counts(raw: Any) -> StepSchedule[tuple[int, ...]]:
    if _is_step_schedule(raw):
        return _parse_step_schedule(
            raw,
            field_name="data.line_counts",
            parse_value=_parse_line_counts_value,
        )
    return StepSchedule(((0, _parse_line_counts_value(raw)),))


def _parse_int_schedule(raw: Any, *, field_name: str) -> StepSchedule[int]:
    return _parse_step_schedule(
        raw,
        field_name=field_name,
        parse_value=lambda value: int(value),
    )


def _parse_variations_plain(raw: Any, *, field_prefix: str) -> RenderVariations:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"{field_prefix} must be a mapping")
    values: dict[str, bool] = {}
    for key in RenderVariations.__dataclass_fields__:
        values[key] = _parse_bool(raw.get(key, False), field_name=f"{field_prefix}.{key}")
    return RenderVariations(**values)


def _parse_single_image_size(raw: Any, *, field_name: str) -> tuple[int, int]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(f"{field_name} must be [width, height]")
    return _parse_image_sizes_value([raw])[0]


def _parse_holdout_render(holdout: dict[str, Any]) -> dict[str, Any]:
    image_size = holdout.get("image_size")
    if image_size is None:
        width = int(holdout["width"])
        height = int(holdout["height"])
        if width < _STEM_STRIDE or height < _STEM_STRIDE:
            raise ValueError(f"holdout size {width}x{height} must be at least {_STEM_STRIDE}x{_STEM_STRIDE}")
        if width % _STEM_STRIDE or height % _STEM_STRIDE:
            raise ValueError(f"holdout size {width}x{height} must be divisible by {_STEM_STRIDE}")
    else:
        width, height = _parse_single_image_size(image_size, field_name="holdout.image_size")
    sentence_len_min = int(holdout["sentence_len_min"])
    sentence_len_max = int(holdout["sentence_len_max"])
    if sentence_len_min < 1:
        raise ValueError("holdout.sentence_len_min must be >= 1")
    if sentence_len_max < sentence_len_min:
        raise ValueError("holdout.sentence_len_max must be >= holdout.sentence_len_min")
    return {
        "width": width,
        "height": height,
        "sentence_len_min": sentence_len_min,
        "sentence_len_max": sentence_len_max,
        "line_counts": _parse_line_counts_value(holdout["line_counts"]),
        "variations": _parse_variations_plain(holdout.get("variations"), field_prefix="holdout.variations"),
    }


def _parse_variations(raw: Any) -> VariationSchedules:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("data.variations must be a mapping")
    schedules: dict[str, StepSchedule[bool]] = {}
    for key in RenderVariations.__dataclass_fields__:
        value = raw.get(key, False)
        schedules[key] = _parse_step_schedule(
            value,
            field_name=f"data.variations.{key}",
            parse_value=lambda item, field=f"data.variations.{key}": _parse_bool(item, field_name=field),
        )
    return VariationSchedules(**schedules)


def _validate_sentence_lengths(
    sentence_len_min: StepSchedule[int],
    sentence_len_max: StepSchedule[int],
) -> None:
    steps = {
        step
        for schedule in (sentence_len_min, sentence_len_max)
        for step, _value in schedule.thresholds
    }
    for step in sorted(steps):
        minimum = sentence_len_min.at(step)
        maximum = sentence_len_max.at(step)
        if minimum < 1:
            raise ValueError("data.sentence_len_min must be >= 1")
        if maximum < minimum:
            raise ValueError(
                f"at step {step}: data.sentence_len_max ({maximum}) "
                f"must be >= data.sentence_len_min ({minimum})"
            )


def _format_step_schedule(name: str, schedule: StepSchedule[Any]) -> str:
    parts = [f"step>={threshold}: {value!r}" for threshold, value in schedule.thresholds]
    return f"{name + ':':<16} {' | '.join(parts)}"


def print_data_schedule(cfg: Config) -> None:
    data = cfg.data
    print(_format_step_schedule("image_sizes", data.image_sizes))
    print(_format_step_schedule("sentence_len_min", data.sentence_len_min))
    print(_format_step_schedule("sentence_len_max", data.sentence_len_max))
    print(_format_step_schedule("line_counts", data.line_counts))
    print("variations:")
    for key in RenderVariations.__dataclass_fields__:
        print(_format_step_schedule(f"  {key}", getattr(data.variations, key)))


def load_config(path: Path | None = None) -> Config:
    config_path = path or CONFIG_PATH
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    base = config_path.parent
    data = raw["data"]
    holdout = raw.get("holdout") or {}
    train = raw["train"]
    infer = raw.get("infer") or {}
    line_counts_raw = data.get("line_counts")
    if line_counts_raw is None:
        min_lines = int(data["min_lines"])
        max_lines = int(data["max_lines"])
        if min_lines < 1:
            raise ValueError("data.min_lines must be >= 1")
        if max_lines < min_lines:
            raise ValueError("data.max_lines must be >= data.min_lines")
        line_counts = StepSchedule(((0, tuple(range(min_lines, max_lines + 1))),))
    else:
        line_counts = _parse_line_counts(line_counts_raw)
    variations = _parse_variations(data.get("variations"))
    sentence_len_min = _parse_int_schedule(data["sentence_len_min"], field_name="data.sentence_len_min")
    sentence_len_max = _parse_int_schedule(data["sentence_len_max"], field_name="data.sentence_len_max")
    _validate_sentence_lengths(sentence_len_min, sentence_len_max)
    holdout_render = _parse_holdout_render(holdout)
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
    return Config(
        max_seq_len=int(raw["max_seq_len"]),
        data=DataSettings(
            seed=int(data["seed"]) if data.get("seed") is not None else None,
            image_sizes=_parse_image_sizes(data.get("image_sizes") or raw.get("image_sizes")),
            sentence_len_min=sentence_len_min,
            sentence_len_max=sentence_len_max,
            line_counts=line_counts,
            variations=variations,
            samples_per_epoch=int(data["samples_per_epoch"]),
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
