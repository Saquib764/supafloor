"""Load SAM predictor from sam_trainer (isolated import)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable

_SAM_ROOT = Path(__file__).resolve().parents[2] / "sam_trainer"


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _with_sam_config(sam_config_path: Path) -> tuple[Any, Any, Callable[..., Any], Callable[..., Any]]:
    """Load sam_trainer model while its `from config import` resolves to sam config."""
    saved: dict[str, Any] = {}
    for key in ("config", "model", "canvas"):
        if key in sys.modules:
            saved[key] = sys.modules[key]

    sam_str = str(_SAM_ROOT.resolve())
    if sam_str not in sys.path:
        sys.path.insert(0, sam_str)

    try:
        sam_config_mod = _load_module("config", _SAM_ROOT / "config.py")
        cfg = sam_config_mod.load_config(sam_config_path)
        sam_model = _load_module("model", _SAM_ROOT / "model.py")
        predictor = sam_model.load_predictor(cfg)
        return predictor, cfg, sam_model.encode_image, sam_model.predict_point_encoded
    finally:
        for key in ("model", "canvas"):
            sys.modules.pop(key, None)
        if "config" in saved:
            sys.modules["config"] = saved["config"]
        elif "config" in sys.modules and sys.modules["config"].__file__ == str(_SAM_ROOT / "config.py"):
            del sys.modules["config"]
        for key, module in saved.items():
            if key != "config" and key not in sys.modules:
                sys.modules[key] = module


def load_sam_predictor(
    sam_config_path: Path,
) -> tuple[Any, Any, Callable[..., Any], Callable[..., Any]]:
    return _with_sam_config(sam_config_path)


def load_viewer_image(path: Path, *, canvas_size: int) -> Any:
    saved_canvas = sys.modules.get("canvas")
    sam_str = str(_SAM_ROOT.resolve())
    if sam_str not in sys.path:
        sys.path.insert(0, sam_str)
    try:
        sam_canvas = _load_module("canvas", _SAM_ROOT / "canvas.py")
        return sam_canvas.load_viewer_image(path, canvas_size=canvas_size, margin=0)
    finally:
        if saved_canvas is not None:
            sys.modules["canvas"] = saved_canvas
        else:
            sys.modules.pop("canvas", None)
