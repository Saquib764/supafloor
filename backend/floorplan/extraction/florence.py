"""Florence-2 room inference via shared florence_trainer inference module."""

from __future__ import annotations

import io
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from extraction.florence_settings import (
    florence_samples_dir,
    load_florence_model_config,
    load_infer_params,
)
from extraction.to_floorplan import rooms_to_floorplan

_TRAINER_ROOT = Path(__file__).resolve().parents[2] / "florence_trainer"
_TRAINER_CONFIG_KEY = "florence_trainer.config"
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
_trainer_imports_ready = False


class FlorenceUnavailableError(RuntimeError):
    pass


def _load_trainer_config_module() -> Any:
    import importlib.util
    import sys

    if _TRAINER_CONFIG_KEY in sys.modules:
        return sys.modules[_TRAINER_CONFIG_KEY]

    config_path = _TRAINER_ROOT / "config.py"
    spec = importlib.util.spec_from_file_location(_TRAINER_CONFIG_KEY, config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load trainer config from {config_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_TRAINER_CONFIG_KEY] = module
    spec.loader.exec_module(module)
    return module


def _ensure_trainer_imports() -> None:
    """Load flat trainer modules once, with trainer config shadowing floorplan's config package."""
    global _trainer_imports_ready
    if _trainer_imports_ready:
        return

    import importlib
    import sys

    trainer_str = str(_TRAINER_ROOT.resolve())
    if trainer_str not in sys.path:
        sys.path.insert(0, trainer_str)

    saved_config = sys.modules.get("config")
    sys.modules["config"] = _load_trainer_config_module()
    try:
        for name in ("data", "model", "inference.progressive", "pre_clean"):
            importlib.import_module(name)
    finally:
        if saved_config is not None:
            sys.modules["config"] = saved_config
        else:
            sys.modules.pop("config", None)

    _trainer_imports_ready = True


@lru_cache(maxsize=1)
def _runtime() -> tuple[Any, Any, Any, Any]:
    _ensure_trainer_imports()
    from model import load_model_and_processor

    cfg = load_florence_model_config()
    params = load_infer_params()
    adapter = cfg.train.session_dir
    if not adapter.exists():
        raise FlorenceUnavailableError(f"LoRA adapter not found: {adapter}")

    model, processor = load_model_and_processor(cfg, adapter=adapter)
    if torch.cuda.is_available():
        model.to("cuda")
    elif torch.backends.mps.is_available():
        model.to("mps")
    model.eval()
    return model, processor, params, cfg


def warmup() -> None:
    """Load and cache the Florence model (call on server startup)."""
    _runtime()


def list_samples() -> list[dict[str, str]]:
    test_dir = florence_samples_dir()
    if not test_dir.is_dir():
        return []
    return [
        {"name": path.name}
        for path in sorted(test_dir.iterdir())
        if path.suffix.lower() in _IMAGE_SUFFIXES
    ]


def sample_image_path(name: str) -> Path:
    if Path(name).name != name or ".." in name:
        raise ValueError("Invalid sample name")
    path = florence_samples_dir() / name
    if not path.is_file() or path.suffix.lower() not in _IMAGE_SUFFIXES:
        raise FileNotFoundError(name)
    return path


def _load_raw_image(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data))


def _image_to_canvas_points(
    points: list[dict[str, float]],
    *,
    image_size: int,
    canvas_size: int,
) -> list[tuple[float, float]]:
    scale = canvas_size / image_size
    return [(float(point["x"]) * scale, float(point["y"]) * scale) for point in points]


def _image_rooms_from_api(rooms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for room in rooms:
        polygon = room.get("polygon") or []
        if len(polygon) < 3:
            continue
        out.append(
            {
                "name": room.get("name") or "Room",
                "polygon": [[int(x), int(y)] for x, y in polygon],
                "count": int(room.get("count") or 1),
            }
        )
    return out


def _rooms_for_response(
    rooms: list[dict[str, Any]],
    *,
    image_size: int,
    canvas_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scale = canvas_size / image_size
    rooms_out: list[dict[str, Any]] = []
    floorplan_rooms: list[dict[str, Any]] = []
    for room in rooms:
        polygon = room.get("polygon") or []
        if len(polygon) < 3:
            continue
        image_polygon = [[int(x), int(y)] for x, y in polygon]
        canvas_polygon = [
            {"x": int(round(x * scale)), "y": int(round(y * scale))}
            for x, y in polygon
        ]
        rooms_out.append(
            {
                "name": room.get("name") or "Room",
                "polygon": image_polygon,
                "count": int(room.get("count") or 1),
            }
        )
        floorplan_rooms.append({"name": room.get("name") or "Room", "polygon": canvas_polygon})

    meta = {"source": "florence", "image_size": image_size, "canvas_size": canvas_size}
    floor_plan = rooms_to_floorplan(
        floorplan_rooms,
        canvas_size=canvas_size,
        meta=meta,
    ).model_dump()
    return rooms_out, floor_plan


def infer_automatic_image(data: bytes) -> dict[str, Any]:
    _ensure_trainer_imports()
    from inference.progressive import infer_automatic

    model, processor, params, _cfg = _runtime()
    raw = _load_raw_image(data)
    print("florence automatic: starting progressive inference", flush=True)
    rooms = infer_automatic(model, processor, params, raw)
    print(f"florence automatic: finished with {len(rooms)} room(s)", flush=True)
    rooms_out, floor_plan = _rooms_for_response(
        rooms,
        image_size=params.image_size,
        canvas_size=params.canvas_size,
    )
    return {
        "rooms": rooms_out,
        "image_size": params.image_size,
        "canvas_size": params.canvas_size,
        "floor_plan": floor_plan,
    }


def infer_points_image(
    data: bytes,
    points: list[dict[str, float]],
    rooms: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not points:
        raise ValueError("At least one point is required")

    _ensure_trainer_imports()
    from data import prepare_image
    from inference.progressive import infer_at_points
    from pre_clean import pre_clean

    model, processor, params, _cfg = _runtime()
    raw = _load_raw_image(data)
    image = prepare_image(pre_clean(raw), params.image_size)
    canvas_points = _image_to_canvas_points(
        points,
        image_size=params.image_size,
        canvas_size=params.canvas_size,
    )
    working = _image_rooms_from_api(rooms or [])
    merged = infer_at_points(
        model,
        processor,
        params,
        image,
        canvas_points,
        working,
    )
    rooms_out, floor_plan = _rooms_for_response(
        merged,
        image_size=params.image_size,
        canvas_size=params.canvas_size,
    )
    return {
        "rooms": rooms_out,
        "image_size": params.image_size,
        "canvas_size": params.canvas_size,
        "floor_plan": floor_plan,
    }
