"""Load inference_core/config.yaml."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from dotenv import load_dotenv

_PACKAGE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = _PACKAGE_DIR / "config.yaml"


@dataclass(frozen=True)
class FpnMaskCleanSettings:
    open_px: int
    close_px: int
    min_area_px: int


@dataclass(frozen=True)
class FpnFloorplanRegionSettings:
    close_px: int
    approx_epsilon: float
    boundary_approx_epsilon: float
    boundary_dilate_px: int
    min_component_area_ratio: float


@dataclass(frozen=True)
class FpnRoomComponentsSettings:
    min_area_px: int
    approx_epsilon: float
    opening_approx_epsilon: float
    bright_vertex_box_radius: int
    bright_vertex_threshold: float
    bright_vertex_reject_fraction: float


RoomNamesBackend = Literal["llm", "paddle"]


@dataclass(frozen=True)
class FpnRoomNamesSettings:
    backend: RoomNamesBackend
    size: int


@dataclass(frozen=True)
class FpnRoomNamesPaddleSettings:
    lang: str
    use_angle_cls: bool
    det_limit_side_len: int
    min_confidence: float
    filter_dimensions: bool
    require_inside_room_mask: bool


@dataclass(frozen=True)
class FpnRoomPolygonLlmSettings:
    size: int
    model: str
    detail: str
    openai_api_key: str
    input_cost_per_million_usd: float
    cached_input_cost_per_million_usd: float
    cache_write_cost_per_million_usd: float
    output_cost_per_million_usd: float


@dataclass(frozen=True)
class FpnSettings:
    size: int
    checkpoint_latest: Path
    checkpoint_old: Path
    output_dir: Path
    mask_clean: FpnMaskCleanSettings
    floorplan_region: FpnFloorplanRegionSettings
    room_components: FpnRoomComponentsSettings
    room_names: FpnRoomNamesSettings
    room_names_llm: FpnRoomPolygonLlmSettings
    room_names_paddle: FpnRoomNamesPaddleSettings
    room_polygon_llm: FpnRoomPolygonLlmSettings


@dataclass(frozen=True)
class ImageEditPrompts:
    clean: str
    openings: str
    furniture: str


@dataclass(frozen=True)
class RoomMaskSettings:
    white_threshold: int
    min_area_px: int
    exclude_largest: bool


@dataclass(frozen=True)
class ImageEditSettings:
    model: str
    quality: str
    aspect_ratio: str
    output_format: str
    background: str
    moderation: str
    number_of_images: int
    output_dir: Path
    prompts: ImageEditPrompts


LabelDataMode = Literal["normal", "llm"]


@dataclass(frozen=True)
class LabelDataSettings:
    mode: LabelDataMode
    input_dir: Path
    output_dir: Path
    pipeline_output_dir: Path


@dataclass(frozen=True)
class EvalUiSettings:
    llm_extract: bool


@dataclass(frozen=True)
class Config:
    canvas_size: int
    hf_cache: Path
    replicate_token: str
    image_edit: ImageEditSettings
    room_masks: RoomMaskSettings
    fpn: FpnSettings
    label_data: LabelDataSettings
    eval_ui: EvalUiSettings
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
            if (parent / "fpn_five_trainer").is_dir():
                return parent
    raise FileNotFoundError("cannot locate backend root (expected fpn_five_trainer/)")


def _resolve(path: Path, base: Path) -> Path:
    return path if path.is_absolute() else (base / path).resolve()


def _load_room_names_settings(fpn: dict[str, Any]) -> FpnRoomNamesSettings:
    room_names = fpn.get("room_names") or {}
    llm = fpn.get("room_names_llm") or {}
    backend = str(room_names.get("backend", "llm"))
    if backend not in ("llm", "paddle"):
        raise ValueError(f"fpn.room_names.backend must be 'llm' or 'paddle', got {backend!r}")
    size = int(room_names.get("size", llm.get("size", 1024)))
    return FpnRoomNamesSettings(backend=cast(RoomNamesBackend, backend), size=size)


def _load_room_names_paddle_settings(raw: dict[str, Any]) -> FpnRoomNamesPaddleSettings:
    return FpnRoomNamesPaddleSettings(
        lang=str(raw.get("lang", "en")),
        use_angle_cls=bool(raw.get("use_angle_cls", False)),
        det_limit_side_len=int(raw.get("det_limit_side_len", 1024)),
        min_confidence=float(raw.get("min_confidence", 0.5)),
        filter_dimensions=bool(raw.get("filter_dimensions", True)),
        require_inside_room_mask=bool(raw.get("require_inside_room_mask", False)),
    )


def _fpn_llm_settings(
    raw: dict[str, Any],
    *,
    default_model: str,
    default_input: float,
    default_cached: float,
    default_cache_write: float,
    default_output: float,
) -> FpnRoomPolygonLlmSettings:
    return FpnRoomPolygonLlmSettings(
        size=int(raw.get("size", 1024)),
        model=str(raw.get("model", default_model)),
        detail=str(raw.get("detail", "high")),
        openai_api_key=str(os.environ.get("OPENAI_API_KEY") or ""),
        input_cost_per_million_usd=float(raw.get("input_cost_per_million_usd", default_input)),
        cached_input_cost_per_million_usd=float(
            raw.get("cached_input_cost_per_million_usd", default_cached)
        ),
        cache_write_cost_per_million_usd=float(
            raw.get("cache_write_cost_per_million_usd", default_cache_write)
        ),
        output_cost_per_million_usd=float(raw.get("output_cost_per_million_usd", default_output)),
    )


def _load_dotenv() -> None:
    load_dotenv(_PACKAGE_DIR / ".env")
    load_dotenv(Path.cwd() / ".env")


def load_config(path: Path | None = None) -> Config:
    _load_dotenv()
    config_path = path or CONFIG_PATH
    if not config_path.is_file():
        config_path = _PACKAGE_DIR / "config.yaml"
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    backend_root = _find_backend_root()
    image_edit = raw.get("image_edit") or {}
    room_masks = raw.get("room_masks") or {}
    fpn = raw.get("fpn") or {}
    label_data = raw.get("label_data") or {}
    eval_ui = raw.get("eval_ui") or {}
    prompts = image_edit.get("prompts") or {}
    inference_root = _find_inference_core_root()
    repo_root = inference_root.parents[1]
    replicate_token = os.environ.get("REPLICATE_API_TOKEN") or str(raw.get("replicate_token") or "")
    mode = str(label_data.get("mode", "normal"))
    if mode not in ("normal", "llm"):
        raise ValueError(f"label_data.mode must be 'normal' or 'llm', got {mode!r}")
    return Config(
        canvas_size=int(raw["canvas_size"]),
        hf_cache=_resolve(Path(raw["hf_cache"]), backend_root),
        replicate_token=replicate_token,
        image_edit=ImageEditSettings(
            model=str(image_edit.get("model", "openai/gpt-image-2.5-flare")),
            quality=str(image_edit.get("quality", "medium")),
            aspect_ratio=str(image_edit.get("aspect_ratio", "1:1")),
            output_format=str(image_edit.get("output_format", "png")),
            background=str(image_edit.get("background", "auto")),
            moderation=str(image_edit.get("moderation", "auto")),
            number_of_images=int(image_edit.get("number_of_images", 1)),
            output_dir=_resolve(
                Path(image_edit.get("output_dir", "output/image_edit")),
                inference_root,
            ),
            prompts=ImageEditPrompts(
                clean=str(
                    prompts.get(
                        "clean",
                        "Change only this: convert the floorplan to CAD-style with filled black walls on a white background. "
                        "Keep walls, doors, windows, and room labels. Remove measurements, legends, scale bars, and other annotations. "
                        "Keep the output aligned with the input — same position, scale, rotation, and canvas framing. "
                        "Do not shift, crop, re-center, rescale, or rotate the plan. Do not change layout, room shapes, wall positions, "
                        "door/window locations, or label text. Do not add or remove rooms, furniture, or decorative elements.",
                    )
                ),
                openings=str(
                    prompts.get(
                        "openings",
                        "Change only this: close door gaps with a green line matching wall thickness. "
                        "Remove door swing arcs and the perpendicular line attached to each door arc (the door-panel line from hinge to swing). "
                        "Close window gaps with a blue line matching wall thickness. "
                        "Do not change walls, room labels, background, layout, alignment, scale, rotation, or canvas framing. "
                        "Do not move, resize, add, or remove doors/windows beyond closing their gaps. "
                        "Do not remove furniture or restyle the drawing.",
                    )
                ),
                furniture=str(
                    prompts.get(
                        "furniture",
                        "Change only this: remove furniture and appliances from the floorplan. "
                        "Remove excess or redundant wall segments — duplicate lines, overlapping walls, and stray wall fragments — "
                        "while keeping the intended room boundary walls intact. "
                        "Do not change doors, windows, room labels, green door lines, blue window lines, background, layout, alignment, "
                        "scale, rotation, or canvas framing. Do not add rooms, furniture, or restyle the drawing.",
                    )
                ),
            ),
        ),
        room_masks=RoomMaskSettings(
            white_threshold=int(room_masks.get("white_threshold", 250)),
            min_area_px=int(room_masks.get("min_area_px", 1000)),
            exclude_largest=bool(room_masks.get("exclude_largest", True)),
        ),
        fpn=FpnSettings(
            size=int(fpn.get("size", 768)),
            checkpoint_latest=_resolve(
                Path(
                    fpn.get(
                        "checkpoint_latest",
                        fpn.get(
                            "checkpoint",
                            "fpn_five_trainer/training_session/last.pt",
                        ),
                    )
                ),
                backend_root,
            ),
            checkpoint_old=_resolve(
                Path(
                    fpn.get(
                        "checkpoint_old",
                        "fpn_five_trainer/training_session_bkp/last.pt",
                    )
                ),
                backend_root,
            ),
            output_dir=_resolve(
                Path(fpn.get("output_dir", "output/fpn")),
                inference_root,
            ),
            mask_clean=FpnMaskCleanSettings(
                open_px=int((fpn.get("mask_clean") or fpn.get("wall_mask") or {}).get("open_px", 2)),
                close_px=int((fpn.get("mask_clean") or fpn.get("wall_mask") or {}).get("close_px", 2)),
                min_area_px=int((fpn.get("mask_clean") or fpn.get("wall_mask") or {}).get("min_area_px", 64)),
            ),
            floorplan_region=FpnFloorplanRegionSettings(
                close_px=int((fpn.get("floorplan_region") or {}).get("close_px", 4)),
                approx_epsilon=float((fpn.get("floorplan_region") or {}).get("approx_epsilon", 2.0)),
                boundary_approx_epsilon=float(
                    (fpn.get("floorplan_region") or {}).get("boundary_approx_epsilon", 8.0)
                ),
                boundary_dilate_px=int(
                    (fpn.get("floorplan_region") or {}).get("boundary_dilate_px", 2)
                ),
                min_component_area_ratio=float(
                    (fpn.get("floorplan_region") or {}).get("min_component_area_ratio", 0.02)
                ),
            ),
            room_components=FpnRoomComponentsSettings(
                min_area_px=int((rc := fpn.get("room_components") or {}).get("min_area_px", 64)),
                approx_epsilon=float(rc.get("approx_epsilon", 2.0)),
                opening_approx_epsilon=float(rc.get("opening_approx_epsilon", 2.0)),
                bright_vertex_box_radius=int(rc.get("bright_vertex_box_radius", 2)),
                bright_vertex_threshold=float(rc.get("bright_vertex_threshold", 200.0)),
                bright_vertex_reject_fraction=float(rc.get("bright_vertex_reject_fraction", 0.6)),
            ),
            room_names=_load_room_names_settings(fpn),
            room_names_llm=_fpn_llm_settings(
                fpn.get("room_names_llm") or {},
                default_model="gpt-5.2",
                default_input=1.75,
                default_cached=0.175,
                default_cache_write=2.1875,
                default_output=14.0,
            ),
            room_names_paddle=_load_room_names_paddle_settings(
                fpn.get("room_names_paddle") or {}
            ),
            room_polygon_llm=_fpn_llm_settings(
                fpn.get("room_polygon_llm") or {},
                default_model="gpt-6-astra",
                default_input=10.0,
                default_cached=1.0,
                default_cache_write=12.5,
                default_output=50.0,
            ),
        ),
        label_data=LabelDataSettings(
            mode=cast(LabelDataMode, mode),
            input_dir=_resolve(
                Path(label_data.get("input_dir", "real_floorplan")),
                inference_root,
            ),
            output_dir=_resolve(
                Path(label_data.get("output_dir", "floorplan-manual-mask")),
                repo_root,
            ),
            pipeline_output_dir=_resolve(
                Path(label_data.get("pipeline_output_dir", "output/label_data")),
                inference_root,
            ),
        ),
        eval_ui=EvalUiSettings(
            llm_extract=bool(eval_ui.get("llm_extract", False)),
        ),
        output=_resolve(
            Path(raw.get("output", "output")),
            inference_root,
        ),
    )
