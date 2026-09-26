"""Load sana_trainer/config.yaml into typed settings."""

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
    split: str
    image_column: str
    caption_column: str
    revision: str | None
    seed: int
    center_crop: bool
    random_flip: bool


@dataclass(frozen=True)
class LoraSettings:
    rank: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]


@dataclass(frozen=True)
class TrainSettings:
    full_finetune: bool
    session_dir: Path
    epochs: int
    batch_size: int
    grad_accum: int
    learning_rate: float
    lr_scheduler: str
    warmup_ratio: float
    weight_decay: float
    max_grad_norm: float
    bf16: bool
    gradient_checkpointing: bool
    weighting_scheme: str
    save_steps: int
    resume: bool
    from_adapter: Path | None
    viz_steps: int
    viz_prompts: tuple[str, ...]
    viz_seed: int
    num_workers: int


@dataclass(frozen=True)
class PreviewSettings:
    prompt: str
    seed: int


@dataclass(frozen=True)
class Config:
    hf_cache: Path
    model_id: str
    variant: str | None
    revision: str | None
    resolution: int
    max_sequence_length: int
    complex_human_instruction: str | None
    data: DataSettings
    lora: LoraSettings
    train: TrainSettings
    preview: PreviewSettings


def _resolve(path: Path, base: Path) -> Path:
    return path if path.is_absolute() else (base / path).resolve()


def _str_list(raw: Any, *, field: str) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{field} must be a non-empty list of strings")
    return tuple(str(item).strip() for item in raw if str(item).strip())


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
    preview = raw.get("preview") or {}
    dataset = str(data["dataset"]).strip()
    if not dataset:
        raise ValueError("data.dataset is required")
    rank = int(lora["rank"])
    alpha = int(lora.get("alpha", rank))
    hf_cache = apply_hf_cache(_resolve(Path(raw["hf_cache"]), base))
    viz_raw = train.get("viz_prompts") or []
    return Config(
        hf_cache=hf_cache,
        model_id=str(raw["model_id"]),
        variant=str(raw["variant"]).strip() if raw.get("variant") else None,
        revision=str(raw["revision"]).strip() if raw.get("revision") else None,
        resolution=int(raw["resolution"]),
        max_sequence_length=int(raw["max_sequence_length"]),
        complex_human_instruction=(
            str(raw["complex_human_instruction"]).strip()
            if raw.get("complex_human_instruction")
            else None
        ),
        data=DataSettings(
            dataset=dataset,
            split=str(data.get("split", "train")),
            image_column=str(data["image_column"]),
            caption_column=str(data["caption_column"]),
            revision=str(data["revision"]).strip() if data.get("revision") else None,
            seed=int(data.get("seed", 42)),
            center_crop=bool(data.get("center_crop", False)),
            random_flip=bool(data.get("random_flip", True)),
        ),
        lora=LoraSettings(
            rank=rank,
            alpha=alpha,
            dropout=float(lora.get("dropout", 0.0)),
            target_modules=_str_list(lora["target_modules"], field="lora.target_modules"),
        ),
        train=TrainSettings(
            full_finetune=bool(train["full_finetune"]),
            session_dir=_resolve(Path(train["session_dir"]), base),
            epochs=int(train["epochs"]),
            batch_size=int(train["batch_size"]),
            grad_accum=int(train["grad_accum"]),
            learning_rate=float(train["learning_rate"]),
            lr_scheduler=str(train.get("lr_scheduler", "constant_with_warmup")),
            warmup_ratio=float(train.get("warmup_ratio", 0.03)),
            weight_decay=float(train.get("weight_decay", 0.01)),
            max_grad_norm=float(train.get("max_grad_norm", 1.0)),
            bf16=bool(train.get("bf16", True)),
            gradient_checkpointing=bool(train.get("gradient_checkpointing", True)),
            weighting_scheme=str(train.get("weighting_scheme", "none")),
            save_steps=int(train["save_steps"]),
            resume=bool(train.get("resume", False)),
            from_adapter=(
                _resolve(Path(raw_adapter), base)
                if (raw_adapter := train.get("from_adapter"))
                else None
            ),
            viz_steps=int(train.get("viz_steps", 100)),
            viz_prompts=_str_list(viz_raw, field="train.viz_prompts") if viz_raw else (),
            viz_seed=int(train.get("viz_seed", 42)),
            num_workers=int(train.get("num_workers", 0)),
        ),
        preview=PreviewSettings(
            prompt=str(preview.get("prompt", "a cute cat")),
            seed=int(preview.get("seed", 42)),
        ),
    )
