"""Load modern_bert_decision/config.yaml into typed settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

import yaml

_PACKAGE_DIR = Path(__file__).resolve().parent


def _project_dir() -> Path:
    if (_PACKAGE_DIR / "pyproject.toml").is_file():
        return _PACKAGE_DIR
    cwd = Path.cwd()
    if (cwd / "config.yaml").is_file():
        return cwd
    return _PACKAGE_DIR


CONFIG_PATH = _project_dir() / "config.yaml"


@dataclass(frozen=True)
class MarkerTokenSettings:
    state: str
    question: str
    option: str


@dataclass(frozen=True)
class InferSettings:
    checkpoint: Path


@dataclass(frozen=True)
class DataSettings:
    dataset: str
    source: str
    race_subset: str
    train_split: str
    val_split: str
    seed: int
    num_workers: int
    option_subset_prob: float
    none_of_these_option: str


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
    init_checkpoint: Path | None
    resume: bool
    save_steps: int
    save_full_encoder: bool
    max_grad_norm: float


@dataclass(frozen=True)
class ModelSettings:
    train_pretrained_encoder: bool = False


def _load_model_settings(raw: dict[str, Any]) -> ModelSettings:
    if "train_pretrained_encoder" in raw:
        return ModelSettings(train_pretrained_encoder=bool(raw["train_pretrained_encoder"]))
    if "freeze_encoder" in raw:
        return ModelSettings(train_pretrained_encoder=not bool(raw["freeze_encoder"]))
    return ModelSettings()


@dataclass(frozen=True)
class DecisionConfig:
    hf_cache: Path
    pretrained_name: str
    model: ModelSettings
    data: DataSettings
    marker_tokens: MarkerTokenSettings
    train: TrainSettings
    infer: InferSettings


def _resolve(path: Path, base: Path) -> Path:
    return path if path.is_absolute() else (base / path).resolve()


def apply_hf_cache(path: Path) -> Path:
    cache = path.resolve()
    hub_cache = cache / "hub"
    cache.mkdir(parents=True, exist_ok=True)
    hub_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    return cache


def hub_cache_dir(cfg: DecisionConfig) -> Path:
    apply_hf_cache(cfg.hf_cache)
    return cfg.hf_cache / "hub"


def hub_from_pretrained_kwargs(cfg: DecisionConfig) -> dict[str, str]:
    return {"cache_dir": str(hub_cache_dir(cfg))}


def datasets_cache_dir(cfg: DecisionConfig) -> Path:
    path = cfg.hf_cache / "datasets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def assert_hub_has_space(cfg: DecisionConfig, *, needed_bytes: int = 650_000_000) -> None:
    import shutil

    hub = hub_cache_dir(cfg)
    _usage = shutil.disk_usage(hub)
    free = _usage.free
    if free < needed_bytes:
        free_mb = free / (1024**2)
        need_mb = needed_bytes / (1024**2)
        raise RuntimeError(
            f"not enough free disk space for Hub download under {hub} "
            f"({free_mb:.0f} MiB free, need ~{need_mb:.0f} MiB for encoder weights). "
            "Free space on this volume or set hf_cache in config.yaml to a path with room."
        )


def load_config(path: Path | None = None) -> DecisionConfig:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    config_path = path or CONFIG_PATH
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    base = config_path.parent
    data_raw = raw.get("data") or {}
    markers_raw = raw.get("marker_tokens") or {}
    infer_raw = raw.get("infer") or {}
    train_raw = raw.get("train") or {}
    model_raw = raw.get("model") or {}
    dataset = str(data_raw.get("dataset") or "").strip()
    if not dataset:
        raise ValueError("data.dataset is required (dataset_manager registry name)")
    source = str(data_raw.get("source") or "auto").strip().lower()
    if source not in ("auto", "local", "hub"):
        raise ValueError("data.source must be auto, local, or hub")
    race_subset = str(data_raw.get("race_subset") or "all").strip().lower()
    if race_subset not in ("all", "middle", "high"):
        raise ValueError("data.race_subset must be all, middle, or high")
    option_subset_prob = float(data_raw.get("option_subset_prob", 0.5))
    if not 0.0 <= option_subset_prob <= 1.0:
        raise ValueError("data.option_subset_prob must be between 0 and 1")
    none_of_these_option = str(data_raw.get("none_of_these_option") or "None of these").strip()
    if not none_of_these_option:
        raise ValueError("data.none_of_these_option must be non-empty")
    pretrained = str(raw.get("pretrained_name") or "").strip()
    if not pretrained:
        raise ValueError("pretrained_name is required")
    return DecisionConfig(
        hf_cache=apply_hf_cache(_resolve(Path(raw.get("hf_cache", "../hf_cache")), base)),
        pretrained_name=pretrained,
        model=_load_model_settings(model_raw),
        data=DataSettings(
            dataset=dataset,
            source=source,
            race_subset=race_subset,
            train_split=str(data_raw.get("train_split") or "train"),
            val_split=str(data_raw.get("val_split") or "validation"),
            seed=int(data_raw.get("seed", 42)),
            num_workers=int(data_raw.get("num_workers", 0)),
            option_subset_prob=option_subset_prob,
            none_of_these_option=none_of_these_option,
        ),
        marker_tokens=MarkerTokenSettings(
            state=str(markers_raw.get("state", "[STATE]")),
            question=str(markers_raw.get("question", "[QUESTION]")),
            option=str(markers_raw.get("option", "[OPTION]")),
        ),
        train=TrainSettings(
            session_dir=_resolve(Path(train_raw.get("session_dir") or "training_session"), base),
            epochs=int(train_raw.get("epochs", 10)),
            batch_size=int(train_raw.get("batch_size", 8)),
            grad_accum=max(1, int(train_raw.get("grad_accum", 1))),
            learning_rate=float(train_raw.get("learning_rate", 2.0e-5)),
            warmup_ratio=float(train_raw.get("warmup_ratio", 0.05)),
            weight_decay=float(train_raw.get("weight_decay", 0.01)),
            bf16=bool(train_raw.get("bf16", False)),
            init_checkpoint=(
                _resolve(Path(init_raw), base)
                if (init_raw := train_raw.get("init_checkpoint"))
                else None
            ),
            resume=bool(train_raw.get("resume", False)),
            save_steps=int(train_raw.get("save_steps", 500)),
            save_full_encoder=bool(train_raw.get("save_full_encoder", False)),
            max_grad_norm=float(train_raw.get("max_grad_norm", 1.0)),
        ),
        infer=InferSettings(
            checkpoint=_resolve(
                Path(infer_raw.get("checkpoint") or f"{train_raw.get('session_dir', 'training_session')}/last"),
                base,
            ),
        ),
    )
