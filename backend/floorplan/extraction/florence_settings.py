"""Floorplan-side Florence inference settings (not trainer config.yaml)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_EXTRACTION_DIR = Path(__file__).resolve().parent
_FLOORPLAN_ROOT = _EXTRACTION_DIR.parent
_CONFIG_PATH = _EXTRACTION_DIR / "florence_config.yaml"


def _resolve(path: Path, base: Path) -> Path:
    return path if path.is_absolute() else (base / path).resolve()


def _ensure_trainer_path() -> None:
    import sys

    trainer_root = (_FLOORPLAN_ROOT.parent / "florence_trainer").resolve()
    trainer_str = str(trainer_root)
    if trainer_str not in sys.path:
        sys.path.append(trainer_str)


def _load_raw_config(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or _CONFIG_PATH
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_infer_params(*, config_path: Path | None = None) -> Any:
    """Load inference params from extraction/florence_config.yaml (or override path)."""
    _ensure_trainer_path()
    from inference.params import infer_params_from_mapping

    return infer_params_from_mapping(_load_raw_config(config_path))


def _trainer_config_module() -> Any:
    """Load florence_trainer/config.py without clashing with floorplan's config package."""
    import importlib.util
    import sys

    name = "florence_trainer.config"
    if name in sys.modules:
        return sys.modules[name]
    path = (_FLOORPLAN_ROOT.parent / "florence_trainer" / "config.py").resolve()
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load trainer config from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_florence_model_config(
    *,
    config_path: Path | None = None,
    adapter_dir: Path | None = None,
) -> Any:
    """Build a minimal trainer Config for model loading only."""
    tc = _trainer_config_module()

    raw = _load_raw_config(config_path)
    base = _FLOORPLAN_ROOT
    hf_cache = tc.apply_hf_cache(_resolve(Path(raw["hf_cache"]), base))
    resolved_adapter = (
        adapter_dir.resolve()
        if adapter_dir is not None
        else _resolve(Path(raw["adapter_dir"]), base)
    )

    return tc.Config(
        hf_cache=hf_cache,
        model_id=str(raw["model_id"]),
        image_size=int(raw["image_size"]),
        canvas_size=int(raw.get("canvas_size") or 1024),
        loc_bins=int(raw["loc_bins"]),
        data=tc.DataSettings(
            dataset="unused",
            max_label_tokens=256,
            empty_polygon_prob=0.0,
            excluded_point_prob=0.0,
            inside_counts=(1,),
            excluded_counts=(1,),
            room_slots=1,
            per_room_variation=1,
        ),
        lora=tc.LoraSettings(dropout=0.05, vision_rank=8, encoder_rank=8, decoder_rank=16),
        train=tc.TrainSettings(
            session_dir=resolved_adapter,
            epochs=1,
            batch_size=1,
            grad_accum=1,
            learning_rate=1e-4,
            warmup_ratio=0.0,
            weight_decay=0.0,
            bf16=bool(raw.get("bf16", True)),
            resume=False,
            from_adapter=None,
            save_steps=1000,
            max_grad_norm=1.0,
            sample_region_size_path=resolved_adapter / "sample_region_size.txt",
            sample_region_refresh_steps=100,
            num_workers=0,
        ),
        infer=tc.InferSettings(
            num_points=16,
            max_new_tokens=int(raw["max_new_tokens"]),
            num_beams=int(raw["num_beams"]),
            seed=int(raw["infer_seed"]) if raw.get("infer_seed") is not None else None,
        ),
    )


def florence_samples_dir() -> Path:
    raw = _load_raw_config()
    return _resolve(Path(raw.get("samples_dir") or "../florence_trainer/test"), _FLOORPLAN_ROOT)
