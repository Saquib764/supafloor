"""Load fpn_trainer/config.yaml into typed settings."""

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
    test_fraction: float
    revision: str | None = None
    seed: int | None = None
    num_workers: int = 0


@dataclass(frozen=True)
class L5TransformerSettings:
    depth: int
    dim: int
    heads: int
    mlp_ratio: float
    level: int
    max_l5_grid_side: int


@dataclass(frozen=True)
class NetSettings:
    stem_channels: int
    backbone_channels: tuple[int, ...]
    depth: int
    fpn_width: int
    out_channels: int
    l5_transformer: L5TransformerSettings | None = None


@dataclass(frozen=True)
class LossSettings:
    focal_alpha: float
    focal_gamma: float
    channel_weights: tuple[float, ...]
    extra_opening_penalty: float


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
    viz_steps: int
    viz_samples: int
    max_grad_norm: float


@dataclass(frozen=True)
class InferSettings:
    size: int
    checkpoint: Path


@dataclass(frozen=True)
class Config:
    hf_cache: Path
    input_sizes: tuple[int, ...]
    data: DataSettings
    net: NetSettings
    loss: LossSettings
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


def _load_l5_transformer(raw: dict[str, Any] | None) -> L5TransformerSettings | None:
    if not raw:
        return None
    dim = int(raw["dim"])
    heads = int(raw["heads"])
    if dim % heads != 0:
        raise ValueError(f"net.l5_transformer.dim ({dim}) must be divisible by heads ({heads})")
    head_dim = dim // heads
    if head_dim % 6 != 0:
        raise ValueError(
            f"net.l5_transformer head_dim ({head_dim}) must be divisible by 6 for 3D RoPE"
        )
    max_l5_grid_side = int(raw["max_l5_grid_side"])
    if max_l5_grid_side < 1:
        raise ValueError(f"net.l5_transformer.max_l5_grid_side ({max_l5_grid_side}) must be ≥ 1")
    return L5TransformerSettings(
        depth=int(raw["depth"]),
        dim=dim,
        heads=heads,
        mlp_ratio=float(raw["mlp_ratio"]),
        level=int(raw["level"]),
        max_l5_grid_side=max_l5_grid_side,
    )


def _parse_input_sizes(raw: Any) -> tuple[int, ...]:
    if isinstance(raw, list):
        sizes = tuple(int(size) for size in raw)
    else:
        sizes = (int(raw),)
    if not sizes:
        raise ValueError("input_size must not be empty")
    for size in sizes:
        if size < 64 or size % 32 != 0:
            raise ValueError(f"input_size ({size}) must be ≥ 64 and divisible by 32")
    return sizes


def _validate_l5_grid_side(input_size: int, l5_transformer: L5TransformerSettings | None) -> None:
    if l5_transformer is None:
        return
    l5_side = input_size // 32
    if l5_side > l5_transformer.max_l5_grid_side:
        raise ValueError(
            f"input_size ({input_size}) needs L5 grid {l5_side}×{l5_side}, "
            f"but net.l5_transformer.max_l5_grid_side is {l5_transformer.max_l5_grid_side} "
            f"(supports input_size up to {l5_transformer.max_l5_grid_side * 32})"
        )


def _load_net_settings(raw: dict[str, Any]) -> NetSettings:
    out_channels = int(raw.get("out_channels", 3))
    if out_channels < 1:
        raise ValueError(f"net.out_channels ({out_channels}) must be ≥ 1")
    return NetSettings(
        stem_channels=int(raw["stem_channels"]),
        backbone_channels=tuple(int(ch) for ch in raw["backbone_channels"]),
        depth=int(raw["depth"]),
        fpn_width=int(raw["fpn_width"]),
        out_channels=out_channels,
        l5_transformer=_load_l5_transformer(raw.get("l5_transformer")),
    )


def load_config(path: Path | None = None) -> Config:
    _load_dotenv()
    config_path = path or CONFIG_PATH
    load_dotenv(config_path.parent)
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    base = config_path.parent
    data = raw["data"]
    net = raw["net"]
    loss = raw["loss"]
    train = raw["train"]
    infer = raw.get("infer") or {}
    dataset = str(data.get("dataset") or "").strip()
    if not dataset:
        raise ValueError("data.dataset is required (dataset_manager registry name)")
    input_sizes = _parse_input_sizes(raw["input_size"])
    hf_cache = apply_hf_cache(_resolve(Path(raw["hf_cache"]), base))
    net_settings = _load_net_settings(net)
    for input_size in input_sizes:
        _validate_l5_grid_side(input_size, net_settings.l5_transformer)
    channel_weights = tuple(float(w) for w in loss.get("channel_weights") or [1.0, 1.0, 1.0])
    if len(channel_weights) != net_settings.out_channels:
        raise ValueError(
            f"loss.channel_weights length ({len(channel_weights)}) "
            f"must match net.out_channels ({net_settings.out_channels})"
        )
    return Config(
        hf_cache=hf_cache,
        input_sizes=input_sizes,
        data=DataSettings(
            dataset=dataset,
            test_fraction=float(data["test_fraction"]),
            revision=str(data["revision"]).strip() if data.get("revision") else None,
            seed=int(data["seed"]) if data.get("seed") is not None else None,
            num_workers=int(data.get("num_workers", 0)),
        ),
        net=net_settings,
        loss=LossSettings(
            focal_alpha=float(loss["focal_alpha"]),
            focal_gamma=float(loss["focal_gamma"]),
            channel_weights=channel_weights,
            extra_opening_penalty=float(loss.get("extra_opening_penalty", 0.0)),
        ),
        train=TrainSettings(
            session_dir=_resolve(Path(train["session_dir"]), base),
            epochs=int(train["epochs"]),
            batch_size=int(train["batch_size"]),
            grad_accum=max(1, int(train.get("grad_accum", 1))),
            learning_rate=float(train["learning_rate"]),
            warmup_ratio=float(train.get("warmup_ratio", 0.0)),
            weight_decay=float(train["weight_decay"]),
            bf16=bool(train["bf16"]),
            resume=bool(train.get("resume", False)),
            save_steps=int(train["save_steps"]),
            viz_steps=int(train["viz_steps"]),
            viz_samples=int(train["viz_samples"]),
            max_grad_norm=float(train["max_grad_norm"]),
        ),
        infer=InferSettings(
            size=int(infer.get("size") or max(input_sizes)),
            checkpoint=_resolve(Path(infer.get("checkpoint") or "training_session/last.pt"), base),
        ),
    )
