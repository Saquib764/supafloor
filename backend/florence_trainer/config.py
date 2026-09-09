"""Load florence_trainer/config.yaml into typed settings for training CLI."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from registry import ROOT as _DATASET_MANAGER_DIR, load_dotenv

_PACKAGE_DIR = Path(__file__).resolve().parent


def _project_dir() -> Path:
    if (_PACKAGE_DIR / "pyproject.toml").is_file():
        return _PACKAGE_DIR
    cwd = Path.cwd()
    if (cwd / "config.yaml").is_file():
        return cwd
    return _PACKAGE_DIR


CONFIG_PATH = _project_dir() / "config.yaml"


def _load_dotenv() -> None:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    load_dotenv(_project_dir(), _DATASET_MANAGER_DIR, Path.cwd())


@dataclass(frozen=True)
class DataSettings:
    dataset: str
    max_label_tokens: int
    empty_polygon_prob: float
    excluded_point_prob: float
    inside_counts: tuple[int, ...]
    excluded_counts: tuple[int, ...]
    room_slots: int
    per_room_variation: int
    revision: str | None = None
    seed: int | None = None

    @property
    def variations_per_floorplan(self) -> int:
        return self.room_slots * self.per_room_variation


@dataclass(frozen=True)
class LoraSettings:
    dropout: float
    vision_rank: int
    encoder_rank: int
    decoder_rank: int


@dataclass(frozen=True)
class TrainSettings:
    session_dir: Path
    epochs: int
    batch_size: int
    grad_accum: int
    learning_rate: float
    warmup_ratio: float
    weight_decay: float
    bf16: bool
    resume: bool
    from_adapter: Path | None
    save_steps: int
    max_grad_norm: float
    sample_region_size_path: Path
    sample_region_refresh_steps: int
    num_workers: int


@dataclass(frozen=True)
class InferSettings:
    num_points: int
    max_new_tokens: int
    num_beams: int
    seed: int | None = None
    adapter: Path | None = None


@dataclass(frozen=True)
class Config:
    hf_cache: Path
    model_id: str
    image_size: int
    canvas_size: int
    loc_bins: int
    data: DataSettings
    lora: LoraSettings
    train: TrainSettings
    infer: InferSettings


def _resolve(path: Path, base: Path) -> Path:
    return path if path.is_absolute() else (base / path).resolve()


def _int_list(raw_value: Any, *, field: str) -> tuple[int, ...]:
    if not isinstance(raw_value, list) or not raw_value:
        raise ValueError(f"data.{field} must be a non-empty list of integers")
    counts = tuple(int(value) for value in raw_value)
    if any(count < 0 for count in counts):
        raise ValueError(f"data.{field} values must be >= 0")
    return counts


def apply_hf_cache(path: Path) -> Path:
    cache = path.resolve()
    hub_cache = cache / "hub"
    datasets_cache = cache / "datasets"
    cache.mkdir(parents=True, exist_ok=True)
    hub_cache.mkdir(parents=True, exist_ok=True)
    datasets_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"
    os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "0"
    return cache


def hub_cache_dir(cfg: Config) -> Path:
    return cfg.hf_cache / "hub"


def datasets_cache_dir(cfg: Config) -> Path:
    return cfg.hf_cache / "datasets"


def load_config(path: Path | None = None) -> Config:
    _load_dotenv()
    config_path = path or CONFIG_PATH
    load_dotenv(config_path.parent)
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    base = config_path.parent
    data = raw["data"]
    lora = raw["lora"]
    train = raw["train"]
    infer = raw["infer"]
    dataset = str(data["dataset"]).strip()
    if not dataset:
        raise ValueError("data.dataset is required (dataset_manager registry name)")
    room_slots = int(data["room_slots"])
    per_room_variation = int(data["per_room_variation"])
    if room_slots < 1:
        raise ValueError("data.room_slots must be >= 1")
    if per_room_variation < 1:
        raise ValueError("data.per_room_variation must be >= 1")
    hf_cache = apply_hf_cache(_resolve(Path(raw["hf_cache"]), base))
    return Config(
        hf_cache=hf_cache,
        model_id=str(raw["model_id"]),
        image_size=int(raw["image_size"]),
        canvas_size=int(raw["canvas_size"]),
        loc_bins=int(raw["loc_bins"]),
        data=DataSettings(
            dataset=dataset,
            max_label_tokens=int(data["max_label_tokens"]),
            empty_polygon_prob=float(data["empty_polygon_prob"]),
            excluded_point_prob=float(data["excluded_point_prob"]),
            inside_counts=_int_list(data["inside_counts"], field="inside_counts"),
            excluded_counts=_int_list(data["excluded_counts"], field="excluded_counts"),
            room_slots=room_slots,
            per_room_variation=per_room_variation,
            revision=str(data["revision"]).strip() if data.get("revision") else None,
            seed=int(data["seed"]) if data["seed"] is not None else None,
        ),
        lora=LoraSettings(
            dropout=float(lora["dropout"]),
            vision_rank=int(lora["vision_rank"]),
            encoder_rank=int(lora["encoder_rank"]),
            decoder_rank=int(lora["decoder_rank"]),
        ),
        train=TrainSettings(
            session_dir=_resolve(Path(train["session_dir"]), base),
            epochs=int(train["epochs"]),
            batch_size=int(train["batch_size"]),
            grad_accum=int(train["grad_accum"]),
            learning_rate=float(train["learning_rate"]),
            warmup_ratio=float(train["warmup_ratio"]),
            weight_decay=float(train["weight_decay"]),
            bf16=bool(train["bf16"]),
            resume=bool(train["resume"]),
            from_adapter=(
                _resolve(Path(raw_from), base)
                if (raw_from := train.get("from_adapter"))
                else None
            ),
            save_steps=int(train["save_steps"]),
            max_grad_norm=float(train["max_grad_norm"]),
            sample_region_size_path=_resolve(Path(train["sample_region_size_path"]), base),
            sample_region_refresh_steps=int(train["sample_region_refresh_steps"]),
            num_workers=int(train.get("num_workers", 0)),
        ),
        infer=InferSettings(
            num_points=int(infer["num_points"]),
            max_new_tokens=int(infer["max_new_tokens"]),
            num_beams=int(infer["num_beams"]),
            seed=int(infer["seed"]) if infer["seed"] is not None else None,
            adapter=(
                _resolve(Path(infer["adapter"]), base)
                if infer.get("adapter")
                else None
            ),
        ),
    )
