"""Load florence_trainer/config.yaml into typed settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
_ENV_PATH = CONFIG_PATH.parent / ".env"


def _load_dotenv() -> None:
    if not _ENV_PATH.is_file():
        return
    for raw in _ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class DataSettings:
    dataset: str
    train_ratio: float
    seed: int
    max_label_tokens: int
    revision: str | None = None


@dataclass(frozen=True)
class LoraSettings:
    r: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]
    freeze_vision: bool


@dataclass(frozen=True)
class TrainSettings:
    output_dir: Path
    epochs: int
    batch_size: int
    grad_accum: int
    learning_rate: float
    warmup_ratio: float
    weight_decay: float
    bf16: bool
    save_steps: int
    eval_steps: int
    max_grad_norm: float


@dataclass(frozen=True)
class InferSettings:
    max_new_tokens: int
    num_beams: int


@dataclass(frozen=True)
class Config:
    hf_cache: Path
    model_id: str
    task_prompt: str
    image_size: int
    loc_bins: int
    data: DataSettings
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


def hub_cache_dir(cfg: Config) -> Path:
    return cfg.hf_cache / "hub"


def datasets_cache_dir(cfg: Config) -> Path:
    return cfg.hf_cache / "datasets"


def load_config(path: Path | None = None) -> Config:
    _load_dotenv()
    config_path = path or CONFIG_PATH
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    base = config_path.parent
    data = raw["data"]
    lora = raw["lora"]
    train = raw["train"]
    infer = raw["infer"]
    dataset = str(data.get("dataset") or "").strip()
    if not dataset:
        raise ValueError("data.dataset is required (dataset_manager registry name)")
    hf_cache = apply_hf_cache(_resolve(Path(raw.get("hf_cache") or "hf_cache"), base))
    return Config(
        hf_cache=hf_cache,
        model_id=str(raw["model_id"]),
        task_prompt=str(raw["task_prompt"]),
        image_size=int(raw["image_size"]),
        loc_bins=int(raw["loc_bins"]),
        data=DataSettings(
            dataset=dataset,
            train_ratio=float(data["train_ratio"]),
            seed=int(data["seed"]),
            max_label_tokens=int(data["max_label_tokens"]),
            revision=str(data["revision"]).strip() if data.get("revision") else None,
        ),
        lora=LoraSettings(
            r=int(lora["r"]),
            alpha=int(lora["alpha"]),
            dropout=float(lora["dropout"]),
            target_modules=tuple(str(name) for name in lora["target_modules"]),
            freeze_vision=bool(lora["freeze_vision"]),
        ),
        train=TrainSettings(
            output_dir=_resolve(Path(train["output_dir"]), base),
            epochs=int(train["epochs"]),
            batch_size=int(train["batch_size"]),
            grad_accum=int(train["grad_accum"]),
            learning_rate=float(train["learning_rate"]),
            warmup_ratio=float(train["warmup_ratio"]),
            weight_decay=float(train["weight_decay"]),
            bf16=bool(train["bf16"]),
            save_steps=int(train["save_steps"]),
            eval_steps=int(train["eval_steps"]),
            max_grad_norm=float(train["max_grad_norm"]),
        ),
        infer=InferSettings(
            max_new_tokens=int(infer["max_new_tokens"]),
            num_beams=int(infer["num_beams"]),
        ),
    )
