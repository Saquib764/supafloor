"""Load vlm_ocr_trainer/config.yaml into typed settings."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

import yaml

from model import DECODER_MAX_SEQ_LEN, STEM_STRIDE_H, STEM_STRIDE_W, validate_image_size

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

_PACKAGE_DIR = Path(__file__).resolve().parent

T = TypeVar("T")

SeqLenMix = tuple[tuple[int, float], ...]


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
class VariationSettings:
    font_size_min: int
    font_size_max: int
    handwritten_prob: float
    bold_prob: float
    italic_prob: float
    bg_min: int
    bg_max: int
    contrast_min: int
    contrast_max: int
    letter_spacing_prob: float
    letter_spacing_max: float
    line_gap_min: float
    line_gap_max: float
    antialias_prob: float
    blur_prob: float
    blur_radius_min: float
    blur_radius_max: float
    noise_prob: float
    noise_sigma_min: float
    noise_sigma_max: float
    jpeg_prob: float
    jpeg_quality_min: int
    jpeg_quality_max: int
    rotation_prob: float
    rotation_degrees: float
    resize_prob: float
    resize_scale_min: float
    resize_scale_max: float
    topic_words_prob: float
    research_prob: float
    punct_after_content_prob: float
    punct_before_content_prob: float
    number_prob: float
    continue_prob: float


def default_variation_settings() -> VariationSettings:
    return VariationSettings(
        font_size_min=8,
        font_size_max=16,
        handwritten_prob=0.08,
        bold_prob=0.14,
        italic_prob=0.14,
        bg_min=210,
        bg_max=255,
        contrast_min=35,
        contrast_max=120,
        letter_spacing_prob=0.55,
        letter_spacing_max=1.8,
        line_gap_min=0.15,
        line_gap_max=0.55,
        antialias_prob=0.35,
        blur_prob=0.45,
        blur_radius_min=0.25,
        blur_radius_max=0.9,
        noise_prob=0.85,
        noise_sigma_min=1.0,
        noise_sigma_max=5.0,
        jpeg_prob=0.4,
        jpeg_quality_min=35,
        jpeg_quality_max=88,
        rotation_prob=0.5,
        rotation_degrees=0.5,
        resize_prob=0.45,
        resize_scale_min=0.82,
        resize_scale_max=0.96,
        topic_words_prob=0.85,
        research_prob=0.12,
        punct_after_content_prob=0.12,
        punct_before_content_prob=0.08,
        number_prob=0.28,
        continue_prob=0.65,
    )


@dataclass(frozen=True)
class VariationSchedule:
    font_size: StepSchedule[tuple[int, int]]
    handwritten_prob: StepSchedule[float]
    bold_prob: StepSchedule[float]
    italic_prob: StepSchedule[float]
    bg_value: StepSchedule[tuple[int, int]]
    contrast: StepSchedule[tuple[int, int]]
    letter_spacing_prob: StepSchedule[float]
    letter_spacing_max: StepSchedule[float]
    line_gap: StepSchedule[tuple[float, float]]
    antialias_prob: StepSchedule[float]
    blur_prob: StepSchedule[float]
    blur_radius: StepSchedule[tuple[float, float]]
    noise_prob: StepSchedule[float]
    noise_sigma: StepSchedule[tuple[float, float]]
    jpeg_prob: StepSchedule[float]
    jpeg_quality: StepSchedule[tuple[int, int]]
    rotation_prob: StepSchedule[float]
    rotation_degrees: StepSchedule[float]
    resize_prob: StepSchedule[float]
    resize_scale: StepSchedule[tuple[float, float]]
    topic_words_prob: StepSchedule[float]
    research_prob: StepSchedule[float]
    punct_after_content_prob: StepSchedule[float]
    punct_before_content_prob: StepSchedule[float]
    number_prob: StepSchedule[float]
    continue_prob: StepSchedule[float]

    def at(self, step: int) -> VariationSettings:
        font_size_min, font_size_max = self.font_size.at(step)
        bg_min, bg_max = self.bg_value.at(step)
        contrast_min, contrast_max = self.contrast.at(step)
        line_gap_min, line_gap_max = self.line_gap.at(step)
        blur_radius_min, blur_radius_max = self.blur_radius.at(step)
        noise_sigma_min, noise_sigma_max = self.noise_sigma.at(step)
        jpeg_quality_min, jpeg_quality_max = self.jpeg_quality.at(step)
        resize_scale_min, resize_scale_max = self.resize_scale.at(step)
        return VariationSettings(
            font_size_min=font_size_min,
            font_size_max=font_size_max,
            handwritten_prob=self.handwritten_prob.at(step),
            bold_prob=self.bold_prob.at(step),
            italic_prob=self.italic_prob.at(step),
            bg_min=bg_min,
            bg_max=bg_max,
            contrast_min=contrast_min,
            contrast_max=contrast_max,
            letter_spacing_prob=self.letter_spacing_prob.at(step),
            letter_spacing_max=self.letter_spacing_max.at(step),
            line_gap_min=line_gap_min,
            line_gap_max=line_gap_max,
            antialias_prob=self.antialias_prob.at(step),
            blur_prob=self.blur_prob.at(step),
            blur_radius_min=blur_radius_min,
            blur_radius_max=blur_radius_max,
            noise_prob=self.noise_prob.at(step),
            noise_sigma_min=noise_sigma_min,
            noise_sigma_max=noise_sigma_max,
            jpeg_prob=self.jpeg_prob.at(step),
            jpeg_quality_min=jpeg_quality_min,
            jpeg_quality_max=jpeg_quality_max,
            rotation_prob=self.rotation_prob.at(step),
            rotation_degrees=self.rotation_degrees.at(step),
            resize_prob=self.resize_prob.at(step),
            resize_scale_min=resize_scale_min,
            resize_scale_max=resize_scale_max,
            topic_words_prob=self.topic_words_prob.at(step),
            research_prob=self.research_prob.at(step),
            punct_after_content_prob=self.punct_after_content_prob.at(step),
            punct_before_content_prob=self.punct_before_content_prob.at(step),
            number_prob=self.number_prob.at(step),
            continue_prob=self.continue_prob.at(step),
        )

    @property
    def final(self) -> VariationSettings:
        font_size_min, font_size_max = self.font_size.final
        bg_min, bg_max = self.bg_value.final
        contrast_min, contrast_max = self.contrast.final
        line_gap_min, line_gap_max = self.line_gap.final
        blur_radius_min, blur_radius_max = self.blur_radius.final
        noise_sigma_min, noise_sigma_max = self.noise_sigma.final
        jpeg_quality_min, jpeg_quality_max = self.jpeg_quality.final
        resize_scale_min, resize_scale_max = self.resize_scale.final
        return VariationSettings(
            font_size_min=font_size_min,
            font_size_max=font_size_max,
            handwritten_prob=self.handwritten_prob.final,
            bold_prob=self.bold_prob.final,
            italic_prob=self.italic_prob.final,
            bg_min=bg_min,
            bg_max=bg_max,
            contrast_min=contrast_min,
            contrast_max=contrast_max,
            letter_spacing_prob=self.letter_spacing_prob.final,
            letter_spacing_max=self.letter_spacing_max.final,
            line_gap_min=line_gap_min,
            line_gap_max=line_gap_max,
            antialias_prob=self.antialias_prob.final,
            blur_prob=self.blur_prob.final,
            blur_radius_min=blur_radius_min,
            blur_radius_max=blur_radius_max,
            noise_prob=self.noise_prob.final,
            noise_sigma_min=noise_sigma_min,
            noise_sigma_max=noise_sigma_max,
            jpeg_prob=self.jpeg_prob.final,
            jpeg_quality_min=jpeg_quality_min,
            jpeg_quality_max=jpeg_quality_max,
            rotation_prob=self.rotation_prob.final,
            rotation_degrees=self.rotation_degrees.final,
            resize_prob=self.resize_prob.final,
            resize_scale_min=resize_scale_min,
            resize_scale_max=resize_scale_max,
            topic_words_prob=self.topic_words_prob.final,
            research_prob=self.research_prob.final,
            punct_after_content_prob=self.punct_after_content_prob.final,
            punct_before_content_prob=self.punct_before_content_prob.final,
            number_prob=self.number_prob.final,
            continue_prob=self.continue_prob.final,
        )


@dataclass(frozen=True)
class DataAtStep:
    image_sizes: tuple[tuple[int, int], ...]
    variations: VariationSettings


@dataclass(frozen=True)
class DataSettings:
    seed: int | None
    image_sizes: StepSchedule[tuple[tuple[int, int], ...]]
    variations: VariationSchedule
    paragraph_counts: tuple[int, ...]
    paragraph_len_max: int
    samples_per_epoch: int

    def at_step(self, step: int) -> DataAtStep:
        return DataAtStep(
            image_sizes=self.image_sizes.at(step),
            variations=self.variations.at(step),
        )

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
    paragraph_len_max: int
    words: Path


@dataclass(frozen=True)
class BenchmarkSettings:
    max_seq_len: int
    num_samples: int | None
    steps: int
    batch_size: int | None
    save_best: bool


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


@dataclass(frozen=True)
class InferSettings:
    checkpoint: Path


@dataclass(frozen=True)
class Config:
    max_seq_len: StepSchedule[SeqLenMix]
    data: DataSettings
    words: WordsSettings
    holdout: HoldoutSettings
    benchmark: BenchmarkSettings
    train: TrainSettings
    infer: InferSettings

    @property
    def seq_len_cap(self) -> int:
        return max(
            length
            for _step, mix in self.max_seq_len.thresholds
            for length, _weight in mix
        )

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


def _parse_seq_len_value(raw: Any, *, field_name: str) -> int:
    value = int(raw)
    if value < 4:
        raise ValueError(f"{field_name} length must be >= 4")
    if value > DECODER_MAX_SEQ_LEN:
        raise ValueError(
            f"{field_name} length ({value}) exceeds DECODER_MAX_SEQ_LEN ({DECODER_MAX_SEQ_LEN})",
        )
    return value


def _parse_max_seq_len_mix(raw: Any, *, field_name: str) -> SeqLenMix:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{field_name} must be [[length, weight], ...]")
    choices: list[tuple[int, float]] = []
    for entry in raw:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError(
                f"each {field_name} entry must be [length, weight], got {entry!r}",
            )
        length = _parse_seq_len_value(entry[0], field_name=field_name)
        weight = float(entry[1])
        if weight <= 0:
            continue
        choices.append((length, weight))
    if not choices:
        raise ValueError(f"{field_name} must include at least one positive weight")
    total = sum(weight for _length, weight in choices)
    return tuple((length, weight / total) for length, weight in choices)


def _parse_max_seq_len_schedule(raw: Any, *, field_name: str) -> StepSchedule[SeqLenMix]:
    if not _is_step_schedule(raw):
        raise ValueError(
            f"{field_name} must be a step schedule: [[step, [[length, weight], ...]], ...]",
        )
    return _parse_step_schedule(
        raw,
        field_name=field_name,
        parse_value=lambda value: _parse_max_seq_len_mix(value, field_name=field_name),
    )


def sample_max_seq_len(mix: SeqLenMix, rng: random.Random) -> int:
    values, weights = zip(*mix, strict=True)
    return rng.choices(values, weights=weights, k=1)[0]


def _parse_image_sizes(raw: Any) -> StepSchedule[tuple[tuple[int, int], ...]]:
    if _is_image_size_schedule(raw):
        return _parse_step_schedule(
            raw,
            field_name="data.image_sizes",
            parse_value=_parse_image_sizes_value,
        )
    return StepSchedule(((0, _parse_image_sizes_value(raw)),))


def _parse_prob_value(raw: Any, *, field_name: str) -> float:
    value = float(raw)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1")
    return value


def _parse_float_value(raw: Any, *, field_name: str) -> float:
    return float(raw)


def _parse_int_range_value(raw: Any, *, field_name: str) -> tuple[int, int]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(f"{field_name} must be [min, max]")
    lo, hi = int(raw[0]), int(raw[1])
    if lo > hi:
        raise ValueError(f"{field_name} min must be <= max")
    return lo, hi


def _parse_float_range_value(raw: Any, *, field_name: str) -> tuple[float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(f"{field_name} must be [min, max]")
    lo, hi = float(raw[0]), float(raw[1])
    if lo > hi:
        raise ValueError(f"{field_name} min must be <= max")
    return lo, hi


def _parse_font_size_value(raw: Any, *, field_name: str) -> tuple[int, int]:
    lo, hi = _parse_int_range_value(raw, field_name=field_name)
    if lo < 6:
        raise ValueError(f"{field_name} min must be >= 6")
    return lo, hi


def _parse_prob_schedule(
    raw: Any,
    *,
    field_name: str,
    default: float,
) -> StepSchedule[float]:
    if raw is None:
        return StepSchedule(((0, default),))
    return _parse_step_schedule(
        raw,
        field_name=field_name,
        parse_value=lambda value: _parse_prob_value(value, field_name=field_name),
    )


def _parse_float_schedule(
    raw: Any,
    *,
    field_name: str,
    default: float,
) -> StepSchedule[float]:
    if raw is None:
        return StepSchedule(((0, default),))
    return _parse_step_schedule(
        raw,
        field_name=field_name,
        parse_value=lambda value: _parse_float_value(value, field_name=field_name),
    )


def _parse_int_range_schedule(
    raw: Any,
    *,
    field_name: str,
    default: tuple[int, int],
) -> StepSchedule[tuple[int, int]]:
    if raw is None:
        return StepSchedule(((0, default),))
    return _parse_step_schedule(
        raw,
        field_name=field_name,
        parse_value=lambda value: _parse_int_range_value(value, field_name=field_name),
    )


def _parse_float_range_schedule(
    raw: Any,
    *,
    field_name: str,
    default: tuple[float, float],
) -> StepSchedule[tuple[float, float]]:
    if raw is None:
        return StepSchedule(((0, default),))
    return _parse_step_schedule(
        raw,
        field_name=field_name,
        parse_value=lambda value: _parse_float_range_value(value, field_name=field_name),
    )


def _parse_font_size_schedule(
    raw: Any,
    *,
    field_name: str,
    default: tuple[int, int],
) -> StepSchedule[tuple[int, int]]:
    if raw is None:
        return StepSchedule(((0, default),))
    return _parse_step_schedule(
        raw,
        field_name=field_name,
        parse_value=lambda value: _parse_font_size_value(value, field_name=field_name),
    )


def _is_grouped_variation_schedule(raw: Any) -> bool:
    if not _is_step_schedule(raw):
        return False
    _step, value = raw[0]
    return isinstance(value, dict)


def _parse_variations(raw: Any) -> VariationSchedule:
    if _is_grouped_variation_schedule(raw):
        raise ValueError(
            "data.variations must map each setting to its own step schedule; "
            "use { font_size: [[0, [8, 16]], ...], blur_prob: [[0, 0], ...], ... }",
        )
    if raw is not None and not isinstance(raw, dict):
        raise ValueError("data.variations must be a mapping of setting → step schedule")

    fields = raw or {}
    defaults = default_variation_settings()
    prefix = "data.variations"

    return VariationSchedule(
        font_size=_parse_font_size_schedule(
            fields.get("font_size"),
            field_name=f"{prefix}.font_size",
            default=(defaults.font_size_min, defaults.font_size_max),
        ),
        handwritten_prob=_parse_prob_schedule(
            fields.get("handwritten_prob"),
            field_name=f"{prefix}.handwritten_prob",
            default=defaults.handwritten_prob,
        ),
        bold_prob=_parse_prob_schedule(
            fields.get("bold_prob"),
            field_name=f"{prefix}.bold_prob",
            default=defaults.bold_prob,
        ),
        italic_prob=_parse_prob_schedule(
            fields.get("italic_prob"),
            field_name=f"{prefix}.italic_prob",
            default=defaults.italic_prob,
        ),
        bg_value=_parse_int_range_schedule(
            fields.get("bg_value"),
            field_name=f"{prefix}.bg_value",
            default=(defaults.bg_min, defaults.bg_max),
        ),
        contrast=_parse_int_range_schedule(
            fields.get("contrast"),
            field_name=f"{prefix}.contrast",
            default=(defaults.contrast_min, defaults.contrast_max),
        ),
        letter_spacing_prob=_parse_prob_schedule(
            fields.get("letter_spacing_prob"),
            field_name=f"{prefix}.letter_spacing_prob",
            default=defaults.letter_spacing_prob,
        ),
        letter_spacing_max=_parse_float_schedule(
            fields.get("letter_spacing_max"),
            field_name=f"{prefix}.letter_spacing_max",
            default=defaults.letter_spacing_max,
        ),
        line_gap=_parse_float_range_schedule(
            fields.get("line_gap"),
            field_name=f"{prefix}.line_gap",
            default=(defaults.line_gap_min, defaults.line_gap_max),
        ),
        antialias_prob=_parse_prob_schedule(
            fields.get("antialias_prob"),
            field_name=f"{prefix}.antialias_prob",
            default=defaults.antialias_prob,
        ),
        blur_prob=_parse_prob_schedule(
            fields.get("blur_prob"),
            field_name=f"{prefix}.blur_prob",
            default=defaults.blur_prob,
        ),
        blur_radius=_parse_float_range_schedule(
            fields.get("blur_radius"),
            field_name=f"{prefix}.blur_radius",
            default=(defaults.blur_radius_min, defaults.blur_radius_max),
        ),
        noise_prob=_parse_prob_schedule(
            fields.get("noise_prob"),
            field_name=f"{prefix}.noise_prob",
            default=defaults.noise_prob,
        ),
        noise_sigma=_parse_float_range_schedule(
            fields.get("noise_sigma"),
            field_name=f"{prefix}.noise_sigma",
            default=(defaults.noise_sigma_min, defaults.noise_sigma_max),
        ),
        jpeg_prob=_parse_prob_schedule(
            fields.get("jpeg_prob"),
            field_name=f"{prefix}.jpeg_prob",
            default=defaults.jpeg_prob,
        ),
        jpeg_quality=_parse_int_range_schedule(
            fields.get("jpeg_quality"),
            field_name=f"{prefix}.jpeg_quality",
            default=(defaults.jpeg_quality_min, defaults.jpeg_quality_max),
        ),
        rotation_prob=_parse_prob_schedule(
            fields.get("rotation_prob"),
            field_name=f"{prefix}.rotation_prob",
            default=defaults.rotation_prob,
        ),
        rotation_degrees=_parse_float_schedule(
            fields.get("rotation_degrees"),
            field_name=f"{prefix}.rotation_degrees",
            default=defaults.rotation_degrees,
        ),
        resize_prob=_parse_prob_schedule(
            fields.get("resize_prob"),
            field_name=f"{prefix}.resize_prob",
            default=defaults.resize_prob,
        ),
        resize_scale=_parse_float_range_schedule(
            fields.get("resize_scale"),
            field_name=f"{prefix}.resize_scale",
            default=(defaults.resize_scale_min, defaults.resize_scale_max),
        ),
        topic_words_prob=_parse_prob_schedule(
            fields.get("topic_words_prob"),
            field_name=f"{prefix}.topic_words_prob",
            default=defaults.topic_words_prob,
        ),
        research_prob=_parse_prob_schedule(
            fields.get("research_prob"),
            field_name=f"{prefix}.research_prob",
            default=defaults.research_prob,
        ),
        punct_after_content_prob=_parse_prob_schedule(
            fields.get("punct_after_content_prob"),
            field_name=f"{prefix}.punct_after_content_prob",
            default=defaults.punct_after_content_prob,
        ),
        punct_before_content_prob=_parse_prob_schedule(
            fields.get("punct_before_content_prob"),
            field_name=f"{prefix}.punct_before_content_prob",
            default=defaults.punct_before_content_prob,
        ),
        number_prob=_parse_prob_schedule(
            fields.get("number_prob"),
            field_name=f"{prefix}.number_prob",
            default=defaults.number_prob,
        ),
        continue_prob=_parse_prob_schedule(
            fields.get("continue_prob"),
            field_name=f"{prefix}.continue_prob",
            default=defaults.continue_prob,
        ),
    )


def _parse_counts_value(raw: Any, *, field_name: str) -> tuple[int, ...]:
    counts = tuple(int(value) for value in raw)
    if not counts:
        raise ValueError(f"{field_name} must not be empty")
    if any(value < 1 for value in counts):
        raise ValueError(f"each {field_name} entry must be >= 1")
    return counts


def _parse_paragraph_render(raw: dict[str, Any], *, field_prefix: str) -> dict[str, Any]:
    paragraph_len_max = int(raw.get("paragraph_len_max", 400))
    if paragraph_len_max < 1:
        raise ValueError(f"{field_prefix}.paragraph_len_max must be >= 1")
    paragraph_counts = _parse_counts_value(
        raw.get("paragraph_counts", [1, 2, 3]),
        field_name=f"{field_prefix}.paragraph_counts",
    )
    return {
        "paragraph_counts": paragraph_counts,
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


def _format_max_seq_len_schedule(schedule: StepSchedule[SeqLenMix]) -> str:
    parts: list[str] = []
    for threshold, mix in schedule.thresholds:
        weights = ", ".join(f"{length}:{weight:.3g}" for length, weight in mix)
        parts.append(f"step>={threshold}: [{weights}]")
    return f"{'max_seq_len:':<16} {' | '.join(parts)}"


def _format_variation_step(value: VariationSettings) -> str:
    return (
        f"font {value.font_size_min}-{value.font_size_max}px, "
        f"AA {value.antialias_prob:.0%}, blur {value.blur_prob:.0%}, "
        f"noise {value.noise_prob:.0%}, jpeg {value.jpeg_prob:.0%}, "
        f"rot {value.rotation_prob:.0%}, resize {value.resize_prob:.0%}"
    )


def _variation_schedules(schedule: VariationSchedule) -> list[tuple[str, StepSchedule[Any]]]:
    return [
        ("font_size", schedule.font_size),
        ("blur_prob", schedule.blur_prob),
        ("noise_prob", schedule.noise_prob),
        ("jpeg_prob", schedule.jpeg_prob),
        ("rotation_prob", schedule.rotation_prob),
        ("resize_prob", schedule.resize_prob),
        ("research_prob", schedule.research_prob),
    ]


def print_data_schedule(cfg: Config) -> None:
    data = cfg.data
    print(
        f"stem_stride:      W÷{STEM_STRIDE_W}, H÷{STEM_STRIDE_H}  "
        f"(256×256 → 64×32 visual grid)",
    )
    print(_format_step_schedule("image_sizes", data.image_sizes))
    print(f"paragraph_counts: {list(data.paragraph_counts)}")
    print(f"paragraph_len:    up to {data.paragraph_len_max}")
    print(f"variations@0:     {_format_variation_step(data.variations.at(0))}")
    print(f"variations@final: {_format_variation_step(data.variations.final)}")
    for name, field_schedule in _variation_schedules(data.variations):
        if len(field_schedule.thresholds) > 1:
            print(_format_step_schedule(f"  {name}", field_schedule))
    print(_format_max_seq_len_schedule(cfg.max_seq_len))


def load_config(path: Path | None = None) -> Config:
    config_path = path or CONFIG_PATH
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    base = config_path.parent
    data = raw["data"]
    holdout = raw.get("holdout") or {}
    words = raw.get("words") or {}
    benchmark = raw.get("benchmark") or {}
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
    benchmark_max_seq_len = int(benchmark.get("max_seq_len", 384))
    if benchmark_max_seq_len < 1:
        raise ValueError("benchmark.max_seq_len must be >= 1")
    benchmark_steps = int(benchmark.get("steps", 0))
    if benchmark_steps < 0:
        raise ValueError("benchmark.steps must be >= 0")
    benchmark_batch_size_raw = benchmark.get("batch_size")
    benchmark_batch_size = (
        int(benchmark_batch_size_raw) if benchmark_batch_size_raw is not None else None
    )
    if benchmark_batch_size is not None and benchmark_batch_size < 1:
        raise ValueError("benchmark.batch_size must be >= 1")
    benchmark_num_samples_raw = benchmark.get("num_samples")
    benchmark_num_samples = (
        int(benchmark_num_samples_raw) if benchmark_num_samples_raw is not None else None
    )
    if benchmark_num_samples is not None and benchmark_num_samples < 1:
        raise ValueError("benchmark.num_samples must be >= 1")
    num_workers = int(train.get("num_workers", 0))
    if num_workers < 0:
        raise ValueError("train.num_workers must be >= 0")
    prefetch_factor = int(train.get("prefetch_factor", 2))
    if prefetch_factor < 1:
        raise ValueError("train.prefetch_factor must be >= 1")
    prefetch_samples = int(train.get("prefetch_samples", 0))
    if prefetch_samples < 0:
        raise ValueError("train.prefetch_samples must be >= 0")
    max_seq_len = _parse_max_seq_len_schedule(raw["max_seq_len"], field_name="max_seq_len")
    return Config(
        max_seq_len=max_seq_len,
        data=DataSettings(
            seed=int(data["seed"]) if data.get("seed") is not None else None,
            image_sizes=_parse_image_sizes(data.get("image_sizes") or raw.get("image_sizes")),
            variations=_parse_variations(data.get("variations")),
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
        benchmark=BenchmarkSettings(
            max_seq_len=benchmark_max_seq_len,
            num_samples=benchmark_num_samples,
            steps=benchmark_steps,
            batch_size=benchmark_batch_size,
            save_best=bool(benchmark.get("save_best", True)),
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
        ),
        infer=InferSettings(
            checkpoint=_resolve(Path(infer.get("checkpoint") or "training_session/last.pt"), base),
        ),
    )
