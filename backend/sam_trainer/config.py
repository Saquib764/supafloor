"""Load sam_trainer/config.yaml into typed settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

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
    room_slots: int
    per_room_variation: int
    empty_mask_prob: float
    point_distance_alpha: float
    test_fraction: float
    revision: str | None = None
    seed: int | None = None
    num_workers: int = 0

    @property
    def variations_per_floorplan(self) -> int:
        return self.room_slots * self.per_room_variation


@dataclass(frozen=True)
class FinetuneSettings:
    neck: bool
    prompt_encoder: bool
    mask_decoder: bool


@dataclass(frozen=True)
class LoraSettings:
    rank: int
    dropout: float
    stage3: bool
    stage4: bool

    @property
    def enabled(self) -> bool:
        return self.stage3 or self.stage4


@dataclass(frozen=True)
class TrainSettings:
    session_dir: Path
    epochs: int
    batch_size: int
    grad_accum: int
    learning_rate: float
    warmup_ratio: float
    weight_decay: float
    max_grad_norm: float
    save_steps: int
    resume: bool
    from_checkpoint: Path | None
    fp16: bool
    viz_steps: int
    viz_samples: int


@dataclass(frozen=True)
class InferSettings:
    multimask_output: bool
    checkpoint: Path | None = None


@dataclass(frozen=True)
class Config:
    model_id: str
    canvas_size: int
    margin: int
    hf_cache: Path
    data: DataSettings
    finetune: FinetuneSettings
    lora: LoraSettings
    train: TrainSettings
    infer: InferSettings


def _resolve(path: Path, base: Path) -> Path:
    return path if path.is_absolute() else (base / path).resolve()


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


def datasets_cache_dir(cfg: Config) -> Path:
    return cfg.hf_cache / "datasets"


def load_config(path: Path | None = None) -> Config:
    _load_dotenv()
    config_path = path or CONFIG_PATH
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    base = config_path.parent
    data = raw.get("data") or {}
    finetune = raw.get("finetune") or {}
    lora = raw.get("lora") or {}
    train = raw.get("train") or {}
    infer = raw.get("infer") or {}

    dataset = str(data.get("dataset", "")).strip()
    if not dataset:
        raise ValueError("data.dataset is required (dataset_manager registry name)")

    room_slots = int(data.get("room_slots", 1))
    per_room_variation = int(data.get("per_room_variation", 1))
    empty_mask_prob = float(data.get("empty_mask_prob", 0.05))
    point_distance_alpha = float(data.get("point_distance_alpha", 1.5))
    test_fraction = float(data.get("test_fraction", 0.2))
    if room_slots < 1:
        raise ValueError("data.room_slots must be >= 1")
    if per_room_variation < 1:
        raise ValueError("data.per_room_variation must be >= 1")
    if not 0.0 <= empty_mask_prob <= 1.0:
        raise ValueError("data.empty_mask_prob must be between 0 and 1")
    if point_distance_alpha < 0.0:
        raise ValueError("data.point_distance_alpha must be >= 0")

    finetune_settings = FinetuneSettings(
        neck=bool(finetune.get("neck", False)),
        prompt_encoder=bool(finetune.get("prompt_encoder", False)),
        mask_decoder=bool(finetune.get("mask_decoder", False)),
    )
    lora_settings = LoraSettings(
        rank=int(lora.get("rank", 8)),
        dropout=float(lora.get("dropout", 0.05)),
        stage3=bool(lora.get("stage3", False)),
        stage4=bool(lora.get("stage4", False)),
    )
    if not lora_settings.enabled:
        raise ValueError("lora: enable at least one of stage3, stage4")
    if lora_settings.rank < 1:
        raise ValueError("lora.rank must be >= 1")

    num_workers = int(data.get("num_workers", 0))

    hf_cache = apply_hf_cache(_resolve(Path(raw["hf_cache"]), base))
    return Config(
        model_id=str(raw["model_id"]).strip(),
        canvas_size=int(raw["canvas_size"]),
        margin=int(raw["margin"]),
        hf_cache=hf_cache,
        data=DataSettings(
            dataset=dataset,
            room_slots=room_slots,
            per_room_variation=per_room_variation,
            empty_mask_prob=empty_mask_prob,
            point_distance_alpha=point_distance_alpha,
            test_fraction=test_fraction,
            revision=str(data["revision"]).strip() if data.get("revision") else None,
            seed=int(data["seed"]) if data.get("seed") is not None else None,
            num_workers=num_workers,
        ),
        finetune=finetune_settings,
        lora=lora_settings,
        train=TrainSettings(
            session_dir=_resolve(Path(train["session_dir"]), base),
            epochs=int(train["epochs"]),
            batch_size=int(train["batch_size"]),
            grad_accum=int(train["grad_accum"]),
            learning_rate=float(train["learning_rate"]),
            warmup_ratio=float(train["warmup_ratio"]),
            weight_decay=float(train["weight_decay"]),
            max_grad_norm=float(train["max_grad_norm"]),
            save_steps=int(train["save_steps"]),
            resume=bool(train.get("resume", False)),
            from_checkpoint=(
                _resolve(Path(raw_from), base)
                if (raw_from := train.get("from_checkpoint"))
                else None
            ),
            fp16=bool(train.get("fp16", False)),
            viz_steps=int(train.get("viz_steps", 10)),
            viz_samples=int(train.get("viz_samples", 3)),
        ),
        infer=InferSettings(
            multimask_output=bool(infer.get("multimask_output", False)),
            checkpoint=(
                _resolve(Path(infer["checkpoint"]), base)
                if infer.get("checkpoint")
                else None
            ),
        ),
    )


def use_fp16(cfg: Config) -> bool:
    if not cfg.train.fp16:
        return False
    import torch

    if torch.cuda.is_available():
        return True
    mps = getattr(torch.backends, "mps", None)
    return mps is not None and mps.is_available()
