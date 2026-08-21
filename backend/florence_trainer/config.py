"""Load florence_trainer/config.yaml into typed settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


def _load_dotenv_from(env_dir: Path) -> None:
    env_path = env_dir / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def _load_dotenv() -> None:
    _load_dotenv_from(_project_dir())
    _load_dotenv_from(Path.cwd())


@dataclass(frozen=True)
class DataSettings:
    dataset: str
    max_label_tokens: int
    revision: str | None = None
    seed: int | None = None


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
    save_steps: int
    test_infer_steps: int
    test_infer_render: bool
    max_grad_norm: float


@dataclass(frozen=True)
class InferSettings:
    num_points: int
    max_new_tokens: int
    num_beams: int
    seed: int | None = None


@dataclass(frozen=True)
class Config:
    hf_cache: Path
    model_id: str
    task_prompt: str
    image_size: int
    canvas_size: int
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
    _load_dotenv_from(config_path.parent)
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
        canvas_size=int(raw.get("canvas_size") or 1024),
        loc_bins=int(raw["loc_bins"]),
        data=DataSettings(
            dataset=dataset,
            max_label_tokens=int(data["max_label_tokens"]),
            revision=str(data["revision"]).strip() if data.get("revision") else None,
            seed=int(data["seed"]) if data.get("seed") is not None else None,
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
            resume=bool(train.get("resume", False)),
            save_steps=int(train["save_steps"]),
            test_infer_steps=int(train["test_infer_steps"]),
            test_infer_render=bool(train.get("test_infer_render", True)),
            max_grad_norm=float(train["max_grad_norm"]),
        ),
        infer=InferSettings(
            num_points=int(infer["num_points"]),
            max_new_tokens=int(infer["max_new_tokens"]),
            num_beams=int(infer["num_beams"]),
            seed=int(infer["seed"]) if infer.get("seed") is not None else None,
        ),
    )
