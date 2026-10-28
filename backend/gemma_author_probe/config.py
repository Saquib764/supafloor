"""Load config.yaml for the frozen-Qwen3 probe authorship classifier."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from env import hf_hub_token
from labels import NUM_CLASSES

BATCH_SIZE = 18
LORA_BLOCK_NAMES = ("lora_1", "lora_2", "lora_3", "lora_4")


def project_root() -> Path:
    """Source tree when launched from the package dir; the install dir otherwise."""
    seeds = [Path.cwd(), Path(__file__).resolve().parent]
    seen: set[Path] = set()
    for seed in seeds:
        here = seed
        for _ in range(8):
            if here in seen:
                break
            seen.add(here)
            pyproject = here / "pyproject.toml"
            if pyproject.is_file() and (here / "config.yaml").is_file():
                name = pyproject.read_text(encoding="utf-8")
                if 'name = "gemma-author-probe"' in name:
                    return here
            if here.parent == here:
                break
            here = here.parent
    return Path(__file__).resolve().parent


CONFIG_PATH = project_root() / "config.yaml"


@dataclass(frozen=True)
class DataSettings:
    root: Path
    train_csv: str
    test_fraction: float
    seed: int
    num_workers: int


@dataclass(frozen=True)
class OptimizerSettings:
    betas: tuple[float, float]
    eps: float


@dataclass(frozen=True)
class ProbeSettings:
    probe_count: int
    freeze: bool
    lr: float


@dataclass(frozen=True)
class HeadSettings:
    freeze: bool
    lr: float
    dropout: float


@dataclass(frozen=True)
class ContrastSettings:
    weight: float
    temperature: float
    hidden_dim: int
    out_dim: int
    freeze: bool
    lr: float


@dataclass(frozen=True)
class LoraBlockSettings:
    name: str
    enabled: bool
    layers: tuple[int, ...]
    r: int
    alpha: int
    dropout: float
    lr: float


@dataclass(frozen=True)
class LoraSettings:
    blocks: tuple[LoraBlockSettings, ...]


@dataclass(frozen=True)
class TrainSettings:
    session_dir: Path
    init_checkpoint: Path | None
    resume: bool
    epochs: int
    batch_size: int
    warmup_ratio: float
    weight_decay: float
    max_grad_norm: float
    max_steps: int
    save_steps: int
    optimizer: OptimizerSettings


@dataclass(frozen=True)
class Config:
    model_id: str
    hf_cache: Path
    max_len: int
    number_of_authors: int
    tap_layers: tuple[int, ...]
    probe: ProbeSettings
    head: HeadSettings
    contrast: ContrastSettings
    lora: LoraSettings
    data: DataSettings
    train: TrainSettings


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


def hub_cache_dir(cfg: Config) -> Path:
    apply_hf_cache(cfg.hf_cache)
    return cfg.hf_cache / "hub"


def hub_from_pretrained_kwargs(cfg: Config) -> dict[str, str]:
    kw = {"cache_dir": str(hub_cache_dir(cfg))}
    token = hf_hub_token()
    if token:
        kw["token"] = token
    return kw


def _load_probe(raw: dict[str, Any]) -> ProbeSettings:
    if not isinstance(raw, dict):
        raise ValueError("probe must be a mapping")
    probe_count = int(raw.get("probe_count", 10))
    if probe_count < 1:
        raise ValueError("probe.probe_count must be >= 1")
    freeze = bool(raw.get("freeze", False))
    lr = float(raw.get("lr", 1.0e-4))
    if not freeze and lr <= 0.0:
        raise ValueError("probe.lr must be > 0 when probe.freeze is false")
    return ProbeSettings(probe_count=probe_count, freeze=freeze, lr=lr)


def _load_head(raw: dict[str, Any]) -> HeadSettings:
    if not isinstance(raw, dict):
        raise ValueError("head must be a mapping")
    freeze = bool(raw.get("freeze", False))
    lr = float(raw.get("lr", 1.0e-4))
    if not freeze and lr <= 0.0:
        raise ValueError("head.lr must be > 0 when head.freeze is false")
    dropout = float(raw.get("dropout", 0.1))
    if not 0.0 <= dropout < 1.0:
        raise ValueError("head.dropout must be in [0, 1)")
    return HeadSettings(freeze=freeze, lr=lr, dropout=dropout)


def _load_contrast(raw: dict[str, Any]) -> ContrastSettings:
    if not isinstance(raw, dict):
        raise ValueError("contrast must be a mapping")
    weight = float(raw.get("weight", 0.0))
    if weight < 0.0:
        raise ValueError("contrast.weight must be >= 0")
    temperature = float(raw.get("temperature", 0.1))
    hidden_dim = int(raw.get("hidden_dim", 64))
    out_dim = int(raw.get("out_dim", 32))
    if hidden_dim < 1:
        raise ValueError("contrast.hidden_dim must be >= 1")
    if out_dim < 1:
        raise ValueError("contrast.out_dim must be >= 1")
    freeze = bool(raw.get("freeze", False))
    lr = float(raw.get("lr", 1.0e-4))
    if not freeze and lr <= 0.0:
        raise ValueError("contrast.lr must be > 0 when contrast.freeze is false")
    if weight > 0.0 and temperature <= 0.0:
        raise ValueError("contrast.temperature must be > 0 when contrast.weight > 0")
    return ContrastSettings(
        weight=weight,
        temperature=temperature,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        freeze=freeze,
        lr=lr,
    )


def _load_lora_block(name: str, raw: dict[str, Any]) -> LoraBlockSettings:
    if not isinstance(raw, dict):
        raise ValueError(f"lora.{name} must be a mapping")
    enabled = bool(raw.get("enabled", False))
    layers_raw = raw.get("layers") or []
    if not isinstance(layers_raw, list):
        raise ValueError(f"lora.{name}.layers must be a list")
    layers = tuple(int(x) for x in layers_raw)
    r = int(raw.get("r", 8))
    alpha = int(raw.get("alpha", 16))
    dropout = float(raw.get("dropout", 0.05))
    lr = float(raw.get("lr", 1.0e-6))
    if enabled and not layers:
        raise ValueError(f"lora.{name} is enabled but layers is empty")
    if enabled and lr <= 0.0:
        raise ValueError(f"lora.{name}.lr must be > 0 when enabled")
    if dropout < 0.0 or dropout >= 1.0:
        raise ValueError(f"lora.{name}.dropout must be in [0, 1)")
    return LoraBlockSettings(
        name=name,
        enabled=enabled,
        layers=layers,
        r=r,
        alpha=alpha,
        dropout=dropout,
        lr=lr,
    )


def _load_lora(raw: dict[str, Any]) -> LoraSettings:
    if not isinstance(raw, dict):
        raise ValueError("lora must be a mapping")
    blocks = [_load_lora_block(name, raw.get(name) or {}) for name in LORA_BLOCK_NAMES]
    return LoraSettings(blocks=tuple(blocks))


def _load_optimizer(train_raw: dict[str, Any]) -> OptimizerSettings:
    opt = train_raw.get("optimizer") or {}
    if not isinstance(opt, dict):
        raise ValueError("train.optimizer must be a mapping")
    raw_betas = opt.get("betas", [0.9, 0.999])
    if not isinstance(raw_betas, list) or len(raw_betas) != 2:
        raise ValueError("train.optimizer.betas must be a list of two floats")
    return OptimizerSettings(
        betas=(float(raw_betas[0]), float(raw_betas[1])),
        eps=float(opt.get("eps", 1.0e-8)),
    )


def _load_tap_layers(raw: dict[str, Any]) -> tuple[int, ...]:
    taps = raw.get("tap_layers")
    if not isinstance(taps, list) or not taps:
        raise ValueError("tap_layers must be a non-empty list of 1-indexed block ids")
    layers = tuple(int(x) for x in taps)
    if any(layer < 1 for layer in layers):
        raise ValueError("tap_layers must be 1-indexed and >= 1")
    if len(set(layers)) != len(layers):
        raise ValueError("tap_layers must be unique")
    return layers


def load_config(path: Path | None = None) -> Config:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    config_path = (path or CONFIG_PATH).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"expected mapping in {config_path}")
    base = config_path.parent

    model_id = str(raw.get("model_id") or "").strip()
    if not model_id:
        raise ValueError("model_id is required")
    max_len = int(raw.get("max_len", 512))
    if max_len < 2:
        raise ValueError("max_len must be >= 2")

    probe = _load_probe(raw.get("probe") or {})
    if max_len <= probe.probe_count:
        raise ValueError("max_len must be greater than probe.probe_count")

    head = _load_head(raw.get("head") or {})
    contrast = _load_contrast(raw.get("contrast") or {})
    lora = _load_lora(raw.get("lora") or {})

    data_raw = raw.get("data") or {}
    train_raw = raw.get("train") or {}
    if not isinstance(data_raw, dict) or not isinstance(train_raw, dict):
        raise ValueError("data and train must be mappings")

    root_raw = data_raw.get("root")
    if not root_raw:
        raise ValueError("data.root is required")
    test_fraction = float(data_raw.get("test_fraction", 0.1))
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError("data.test_fraction must be in [0, 1)")

    number_of_authors = int(raw.get("number_of_authors", 0))
    if number_of_authors < 1:
        raise ValueError("number_of_authors must be >= 1")
    if number_of_authors > NUM_CLASSES:
        raise ValueError(
            f"number_of_authors is {number_of_authors}; each prompt has {NUM_CLASSES} models"
        )

    batch_size = int(train_raw.get("batch_size", BATCH_SIZE))
    if batch_size < 1:
        raise ValueError("train.batch_size must be >= 1")
    if batch_size % number_of_authors != 0:
        raise ValueError("train.batch_size must be divisible by number_of_authors")
    prompts_per_batch = batch_size // number_of_authors
    if contrast.weight > 0.0:
        if number_of_authors < 2:
            raise ValueError("contrast.weight > 0 requires number_of_authors >= 2")
        if prompts_per_batch < 2:
            raise ValueError(
                "contrast.weight > 0 requires train.batch_size / number_of_authors >= 2"
            )

    epochs = int(train_raw.get("epochs", 3))
    if epochs < 1:
        raise ValueError("train.epochs must be >= 1")
    max_steps = int(train_raw.get("max_steps", 0))
    if max_steps < 0:
        raise ValueError("train.max_steps must be >= 0")
    save_steps = int(train_raw.get("save_steps", 100))
    if save_steps < 1:
        raise ValueError("train.save_steps must be >= 1")

    checkpoint_raw = train_raw.get("checkpoint") or {}
    if not isinstance(checkpoint_raw, dict):
        raise ValueError("train.checkpoint must be a mapping")
    init_raw = checkpoint_raw.get("init")
    init_checkpoint = _resolve(Path(str(init_raw)), base) if init_raw else None
    resume = bool(checkpoint_raw.get("resume", False))

    return Config(
        model_id=model_id,
        hf_cache=apply_hf_cache(_resolve(Path(raw.get("hf_cache") or "../hf_cache"), base)),
        max_len=max_len,
        number_of_authors=number_of_authors,
        tap_layers=_load_tap_layers(raw),
        probe=probe,
        head=head,
        contrast=contrast,
        lora=lora,
        data=DataSettings(
            root=_resolve(Path(str(root_raw)), base),
            train_csv=str(data_raw.get("train_csv") or "train.csv"),
            test_fraction=test_fraction,
            seed=int(data_raw.get("seed", 42)),
            num_workers=int(data_raw.get("num_workers", 0)),
        ),
        train=TrainSettings(
            session_dir=_resolve(Path(train_raw.get("session_dir") or "training_session"), base),
            init_checkpoint=init_checkpoint,
            resume=resume,
            epochs=epochs,
            batch_size=batch_size,
            warmup_ratio=float(train_raw.get("warmup_ratio", 0.05)),
            weight_decay=float(train_raw.get("weight_decay", 0.01)),
            max_grad_norm=float(train_raw.get("max_grad_norm", 1.0)),
            max_steps=max_steps,
            save_steps=save_steps,
            optimizer=_load_optimizer(train_raw),
        ),
    )
