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

ARCHITECTURE_SEGMENT_MARKERS = "segment_markers_v1"
# Legacy checkpoints saved under this id used CLS+[SEP] layout without segment markers.
ARCHITECTURE_CLS_PER_CANDIDATE = "cls_per_candidate_v1"


def _project_dir() -> Path:
    if (_PACKAGE_DIR / "pyproject.toml").is_file():
        return _PACKAGE_DIR
    cwd = Path.cwd()
    if (cwd / "config.yaml").is_file():
        return cwd
    return _PACKAGE_DIR


CONFIG_PATH = _project_dir() / "config.yaml"


@dataclass(frozen=True)
class PromptSettings:
    candidate_question: str


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


@dataclass(frozen=True)
class TrainSettings:
    session_dir: Path
    epochs: int
    batch_size: int
    grad_accum: int
    head_learning_rate: float
    train_cls_embedding: bool
    cls_embedding_learning_rate: float
    marker_embedding_train: tuple[bool, ...]
    marker_embedding_learning_rates: tuple[float, ...]
    encoder_layer_train: tuple[bool, ...]
    encoder_layer_learning_rates: tuple[float, ...]
    warmup_ratio: float
    weight_decay: float
    bf16: bool
    init_checkpoint: Path | None
    resume: bool
    save_steps: int
    max_grad_norm: float
    eval_samples: int
    eval_steps: int
    eval_seed: int
    test_mode: bool
    test_samples: int
    variations: int


@dataclass(frozen=True)
class ModelSettings:
    finetune_last_n_layers: int = 0
    score_head_dropout: float = 0.0


def _load_model_settings(
    raw: dict[str, Any],
    *,
    finetune_last_n_layers: int,
) -> ModelSettings:
    if bool(raw.get("train_pretrained_encoder", False)):
        raise ValueError(
            "model.train_pretrained_encoder is removed (no full fine-tune); use train.encoder"
        )
    if raw.get("freeze_encoder") is False:
        raise ValueError(
            "freeze_encoder: false implied full fine-tune; use train.encoder.last_n_layers"
        )
    if raw.get("encoder_layer_learning_rates") or raw.get("finetune_last_n_layers"):
        raise ValueError(
            "move finetune_last_n_layers and encoder layer LRs to train.encoder in config.yaml"
        )
    if raw.get("cls_embedding") or raw.get("train_cls_embedding") is not None:
        raise ValueError("move cls_embedding settings to train.cls_embedding in config.yaml")
    score_head = raw.get("score_head") or {}
    dropout = float(score_head.get("dropout", raw.get("score_head_dropout", 0.0)))
    if not 0.0 <= dropout < 1.0:
        raise ValueError("model.score_head.dropout must be in [0.0, 1.0)")
    return ModelSettings(
        finetune_last_n_layers=finetune_last_n_layers,
        score_head_dropout=dropout,
    )


def _load_cls_embedding_settings(
    train_raw: dict[str, Any],
    lr_raw: dict[str, Any],
) -> tuple[bool, float]:
    cls = train_raw.get("cls_embedding") or {}
    if isinstance(cls, dict) and cls:
        train = bool(cls.get("train", False))
        lr = float(cls.get("lr", lr_raw.get("cls_embedding", 1.0e-4)))
        return train, lr
    train = bool(train_raw.get("train_cls_embedding", False))
    lr = float(lr_raw.get("cls_embedding", train_raw.get("cls_embedding_learning_rate", 1.0e-4)))
    return train, lr


def _finetune_last_n_layers(model_raw: dict[str, Any], train_raw: dict[str, Any]) -> int:
    encoder = train_raw.get("encoder") or {}
    if encoder.get("last_n_layers") is not None:
        return max(0, int(encoder["last_n_layers"]))
    if model_raw.get("finetune_last_n_layers") is not None:
        return max(0, int(model_raw["finetune_last_n_layers"]))
    return 0


def _load_marker_embedding_settings(
    train_raw: dict[str, Any],
) -> tuple[tuple[bool, ...], tuple[float, ...]]:
    from model.segment_markers import MARKER_NAMES

    markers = train_raw.get("markers")
    if isinstance(markers, dict) and markers:
        train_flags: list[bool] = []
        layer_lrs: list[float] = []
        for name in MARKER_NAMES:
            entry = markers.get(name)
            if not isinstance(entry, dict):
                raise ValueError(f"train.markers.{name} must be {{train, lr}}")
            train_flags.append(bool(entry.get("train", True)))
            layer_lrs.append(float(entry.get("lr", 1.0e-5)))
        return tuple(train_flags), tuple(layer_lrs)

    n = len(MARKER_NAMES)
    train_raw_flags = train_raw.get("marker_embedding_train")
    rates_raw = train_raw.get("marker_embedding_learning_rates")
    if train_raw_flags is None:
        train_flags_t = tuple(True for _ in range(n))
    elif not isinstance(train_raw_flags, list) or len(train_raw_flags) != n:
        raise ValueError(f"train.marker_embedding_train must be a list of {n} booleans")
    else:
        train_flags_t = tuple(bool(x) for x in train_raw_flags)
    if rates_raw is None:
        layer_lrs_t = tuple(1.0e-5 for _ in range(n))
    elif not isinstance(rates_raw, list) or len(rates_raw) != n:
        raise ValueError(f"train.marker_embedding_learning_rates must be a list of length {n}")
    else:
        layer_lrs_t = tuple(float(x) for x in rates_raw)
    return train_flags_t, layer_lrs_t


def _load_encoder_layer_finetune(
    train_raw: dict[str, Any],
    *,
    finetune_last_n_layers: int,
) -> tuple[tuple[bool, ...], tuple[float, ...]]:
    n = finetune_last_n_layers
    encoder = train_raw.get("encoder") or {}
    layers = encoder.get("layers")
    if isinstance(layers, list) and layers:
        if n <= 0:
            raise ValueError("train.encoder.layers requires train.encoder.last_n_layers > 0")
        if len(layers) != n:
            raise ValueError(
                f"train.encoder.layers length {len(layers)} != train.encoder.last_n_layers ({n})"
            )
        layer_train: list[bool] = []
        layer_lrs: list[float] = []
        for entry in layers:
            if not isinstance(entry, dict):
                raise ValueError("each train.encoder.layers item must be {train, lr}")
            layer_train.append(bool(entry.get("train", True)))
            layer_lrs.append(float(entry.get("lr", 1.0e-5)))
        return tuple(layer_train), tuple(layer_lrs)

    train_raw_flags = train_raw.get("encoder_layer_train")
    rates_raw = train_raw.get("encoder_layer_learning_rates")
    if n <= 0:
        if train_raw_flags or rates_raw:
            raise ValueError(
                "train.encoder_layer_train / encoder_layer_learning_rates require "
                "train.encoder.last_n_layers > 0"
            )
        return (), ()
    if not isinstance(rates_raw, list) or len(rates_raw) != n:
        raise ValueError(
            f"train.encoder_layer_learning_rates must be a list of length "
            f"train.encoder.last_n_layers ({n})"
        )
    layer_lrs_t = tuple(float(x) for x in rates_raw)
    if train_raw_flags is None:
        layer_train_t = tuple(True for _ in range(n))
    elif not isinstance(train_raw_flags, list) or len(train_raw_flags) != n:
        raise ValueError(
            f"train.encoder_layer_train must be a list of {n} booleans "
            f"(train.encoder.last_n_layers)"
        )
    else:
        layer_train_t = tuple(bool(x) for x in train_raw_flags)
    return layer_train_t, layer_lrs_t


def _train_get(train_raw: dict[str, Any], *path: str, legacy_key: str | None = None, default: Any = None) -> Any:
    node: Any = train_raw
    for key in path:
        if not isinstance(node, dict):
            return default
        node = node.get(key)
        if node is None:
            break
    if node is not None:
        return node
    if legacy_key is not None:
        return train_raw.get(legacy_key, default)
    return default


@dataclass(frozen=True)
class DecisionConfig:
    hf_cache: Path
    pretrained_name: str
    model: ModelSettings
    data: DataSettings
    prompt: PromptSettings
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
    prompt_raw = raw.get("prompt") or {}
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
    candidate_question = str(
        prompt_raw.get("candidate_question") or "Is the correct answer {candidate}?"
    ).strip()
    if "{candidate}" not in candidate_question:
        raise ValueError("prompt.candidate_question must contain {candidate}")
    hub_raw = raw.get("hub") or {}
    pretrained = str(hub_raw.get("pretrained") or raw.get("pretrained_name") or "").strip()
    if not pretrained:
        raise ValueError("hub.pretrained (or pretrained_name) is required")
    hf_cache_path = hub_raw.get("cache") or raw.get("hf_cache") or "../hf_cache"
    finetune_n = _finetune_last_n_layers(model_raw, train_raw)
    model = _load_model_settings(model_raw, finetune_last_n_layers=finetune_n)
    marker_train, marker_lrs = _load_marker_embedding_settings(train_raw)
    encoder_layer_train, encoder_layer_lrs = _load_encoder_layer_finetune(
        train_raw,
        finetune_last_n_layers=finetune_n,
    )
    lr_raw = train_raw.get("learning_rates") or {}
    train_cls, cls_lr = _load_cls_embedding_settings(train_raw, lr_raw)
    return DecisionConfig(
        hf_cache=apply_hf_cache(_resolve(Path(hf_cache_path), base)),
        pretrained_name=pretrained,
        model=model,
        data=DataSettings(
            dataset=dataset,
            source=source,
            race_subset=race_subset,
            train_split=str(data_raw.get("train_split") or "train"),
            val_split=str(data_raw.get("val_split") or "validation"),
            seed=int(data_raw.get("seed", 42)),
            num_workers=int(data_raw.get("num_workers", 0)),
        ),
        prompt=PromptSettings(candidate_question=candidate_question),
        train=TrainSettings(
            session_dir=_resolve(
                Path(_train_get(train_raw, "session_dir", default="training_session")),
                base,
            ),
            epochs=int(_train_get(train_raw, "schedule", "epochs", legacy_key="epochs", default=10)),
            batch_size=int(_train_get(train_raw, "schedule", "batch_size", legacy_key="batch_size", default=8)),
            grad_accum=max(
                1,
                int(_train_get(train_raw, "schedule", "grad_accum", legacy_key="grad_accum", default=1)),
            ),
            head_learning_rate=float(
                lr_raw.get("head", train_raw.get("head_learning_rate", 2.0e-5))
            ),
            train_cls_embedding=train_cls,
            cls_embedding_learning_rate=cls_lr,
            marker_embedding_train=marker_train,
            marker_embedding_learning_rates=marker_lrs,
            encoder_layer_train=encoder_layer_train,
            encoder_layer_learning_rates=encoder_layer_lrs,
            warmup_ratio=float(
                _train_get(train_raw, "schedule", "warmup_ratio", legacy_key="warmup_ratio", default=0.05)
            ),
            weight_decay=float(
                _train_get(train_raw, "schedule", "weight_decay", legacy_key="weight_decay", default=0.01)
            ),
            bf16=bool(_train_get(train_raw, "schedule", "bf16", legacy_key="bf16", default=False)),
            init_checkpoint=(
                _resolve(Path(init_raw), base)
                if (
                    init_raw := _train_get(
                        train_raw, "checkpoint", "init", legacy_key="init_checkpoint"
                    )
                )
                else None
            ),
            resume=bool(_train_get(train_raw, "checkpoint", "resume", legacy_key="resume", default=False)),
            save_steps=int(
                _train_get(train_raw, "checkpoint", "save_steps", legacy_key="save_steps", default=500)
            ),
            max_grad_norm=float(
                _train_get(
                    train_raw, "schedule", "max_grad_norm", legacy_key="max_grad_norm", default=1.0
                )
            ),
            eval_samples=max(
                1,
                int(_train_get(train_raw, "eval", "samples", legacy_key="eval_samples", default=100)),
            ),
            eval_steps=max(
                1,
                int(
                    _train_get(
                        train_raw,
                        "eval",
                        "steps",
                        legacy_key="eval_steps",
                        default=_train_get(
                            train_raw, "checkpoint", "save_steps", legacy_key="save_steps", default=500
                        ),
                    )
                ),
            ),
            eval_seed=int(
                _train_get(
                    train_raw,
                    "eval",
                    "seed",
                    legacy_key="eval_seed",
                    default=data_raw.get("seed", 42),
                )
            ),
            test_mode=bool(
                _train_get(train_raw, "test_mode", "enabled", legacy_key="test_mode", default=False)
            ),
            test_samples=max(
                1,
                int(_train_get(train_raw, "test_mode", "samples", legacy_key="test_samples", default=200)),
            ),
            variations=max(
                1,
                int(_train_get(train_raw, "schedule", "variations", legacy_key="variations", default=2)),
            ),
        ),
        infer=InferSettings(
            checkpoint=_resolve(
                Path(infer_raw.get("checkpoint") or f"{train_raw.get('session_dir', 'training_session')}/last"),
                base,
            ),
        ),
    )
