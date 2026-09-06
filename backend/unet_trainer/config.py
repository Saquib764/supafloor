"""Load unet_trainer/config.yaml into typed settings."""

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
    load_dotenv(_project_dir(), _DATASET_MANAGER_DIR, Path.cwd())


@dataclass(frozen=True)
class DataSettings:
    dataset: str
    revision: str | None = None
    seed: int | None = None


@dataclass(frozen=True)
class TaesdSettings:
    encoder: Path
    decoder: Path


@dataclass(frozen=True)
class VitSettings:
    dim: int
    depth: int
    heads: int
    mlp_ratio: float
    patch_size: int


@dataclass(frozen=True)
class LossSettings:
    ink_weight: float


@dataclass(frozen=True)
class TrainSettings:
    session_dir: Path
    epochs: int
    batch_size: int
    grad_accum: int
    learning_rate: float
    weight_decay: float
    bf16: bool
    resume: bool
    save_steps: int
    recon_steps: int
    max_grad_norm: float
    num_workers: int
    precompute_latent: bool


@dataclass(frozen=True)
class Config:
    hf_cache: Path
    image_size: int
    data: DataSettings
    taesd: TaesdSettings
    vit: VitSettings
    loss: LossSettings
    train: TrainSettings

    @property
    def compressed_size(self) -> int:
        from model import compressed_hw

        return compressed_hw(self.image_size, self.vit.patch_size)


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
    load_dotenv(config_path.parent)
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    base = config_path.parent
    data = raw["data"]
    taesd = raw["taesd"]
    vit = raw["vit"]
    loss = raw["loss"]
    train = raw["train"]
    dataset = str(data.get("dataset") or "").strip()
    if not dataset:
        raise ValueError("data.dataset is required (dataset_manager registry name)")
    image_size = int(raw["image_size"])
    patch_size = int(vit["patch_size"])
    from model import compressed_hw

    compressed_hw(image_size, patch_size)
    hf_cache = apply_hf_cache(_resolve(Path(raw["hf_cache"]), base))
    return Config(
        hf_cache=hf_cache,
        image_size=image_size,
        data=DataSettings(
            dataset=dataset,
            revision=str(data["revision"]).strip() if data.get("revision") else None,
            seed=int(data["seed"]) if data.get("seed") is not None else None,
        ),
        taesd=TaesdSettings(
            encoder=_resolve(Path(taesd["encoder"]), base),
            decoder=_resolve(Path(taesd["decoder"]), base),
        ),
        vit=VitSettings(
            dim=int(vit["dim"]),
            depth=int(vit["depth"]),
            heads=int(vit["heads"]),
            mlp_ratio=float(vit["mlp_ratio"]),
            patch_size=patch_size,
        ),
        loss=LossSettings(ink_weight=float(loss["ink_weight"])),
        train=TrainSettings(
            session_dir=_resolve(Path(train["session_dir"]), base),
            epochs=int(train["epochs"]),
            batch_size=int(train["batch_size"]),
            grad_accum=max(1, int(train.get("grad_accum", 1))),
            learning_rate=float(train["learning_rate"]),
            weight_decay=float(train["weight_decay"]),
            bf16=bool(train["bf16"]),
            resume=bool(train.get("resume", False)),
            save_steps=int(train["save_steps"]),
            recon_steps=int(train["recon_steps"]),
            max_grad_norm=float(train["max_grad_norm"]),
            num_workers=int(train.get("num_workers", 0)),
            precompute_latent=bool(train.get("precompute_latent", True)),
        ),
    )
