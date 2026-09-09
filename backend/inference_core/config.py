"""Load inference_core/config.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_PACKAGE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = _PACKAGE_DIR / "config.yaml"


@dataclass(frozen=True)
class FitSettings:
    min_sides: int
    max_sides: int
    output_dir: Path


@dataclass(frozen=True)
class GridSettings:
    size: int
    tries_per_cell: int
    seed: int | None


@dataclass(frozen=True)
class PointFilterSettings:
    dilate_px: int


@dataclass(frozen=True)
class MaskSmoothSettings:
    close_px: int
    smooth_sigma: float
    approx_epsilon: float


@dataclass(frozen=True)
class Config:
    canvas_size: int
    margin: int
    hf_cache: Path
    sam_config: Path
    grid: GridSettings
    point_filter: PointFilterSettings
    fit: FitSettings
    mask_smooth: MaskSmoothSettings
    output: Path


def _find_inference_core_root() -> Path:
    for start in (Path(__file__).resolve().parent, Path.cwd()):
        for parent in (start, *start.parents):
            if (parent / "util").is_dir() and (parent / "config.yaml").is_file():
                return parent
    return _PACKAGE_DIR


def _find_backend_root() -> Path:
    for start in (Path(__file__).resolve().parent, Path.cwd()):
        for parent in (start, *start.parents):
            if (parent / "sam_trainer" / "config.yaml").is_file():
                return parent
    raise FileNotFoundError("cannot locate backend/sam_trainer/config.yaml")


def _resolve(path: Path, base: Path) -> Path:
    return path if path.is_absolute() else (base / path).resolve()


def load_config(path: Path | None = None) -> Config:
    config_path = path or CONFIG_PATH
    if not config_path.is_file():
        config_path = _PACKAGE_DIR / "config.yaml"
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    backend_root = _find_backend_root()
    grid = raw.get("grid") or raw.get("progressive") or {}
    point_filter = raw.get("point_filter") or {}
    fit = raw.get("fit") or {}
    mask_smooth = raw.get("mask_smooth") or {}
    sam_config = backend_root / "sam_trainer" / "config.yaml"
    return Config(
        canvas_size=int(raw["canvas_size"]),
        margin=int(raw["margin"]),
        hf_cache=_resolve(Path(raw["hf_cache"]), backend_root),
        sam_config=sam_config,
        grid=GridSettings(
            size=int(grid.get("size", grid.get("grid_size", 100))),
            tries_per_cell=int(grid.get("tries_per_cell", grid.get("grid_tries_per_cell", 5))),
            seed=int(grid["seed"]) if grid.get("seed") is not None else None,
        ),
        point_filter=PointFilterSettings(
            dilate_px=int(point_filter.get("dilate_px", 24)),
        ),
        fit=FitSettings(
            min_sides=int(fit.get("min_sides", 4)),
            max_sides=int(fit.get("max_sides", 10)),
            output_dir=_resolve(
                Path(fit.get("output_dir", "output/fit")),
                _find_inference_core_root(),
            ),
        ),
        mask_smooth=MaskSmoothSettings(
            close_px=int(mask_smooth.get("close_px", 2)),
            smooth_sigma=float(mask_smooth.get("smooth_sigma", 1.0)),
            approx_epsilon=float(mask_smooth.get("approx_epsilon", 2.0)),
        ),
        output=_resolve(
            Path(raw.get("output", "output")),
            _find_inference_core_root(),
        ),
    )
