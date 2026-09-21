"""LLM floor plan polygon refinement from input + combined room mask."""

from __future__ import annotations

import base64
import colorsys
import json
import mimetypes
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from openai import OpenAI
from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, Field

from util.final_mask_polygons import (
    PolygonRing,
    load_layer_polygons,
    save_final_mask_contour_polygons,
)
from util.render import floorplan_view, merge_panels, with_label
from util.room_components import room_components_from_mask

DetailLevel = Literal["low", "high", "auto"]

LlmTokenUsage = dict[str, int]
StepLogFn = Callable[[str], None]

SYSTEM_PROMPT = """You are a precise floor plan polygon tracing system.
Image size is exactly {width} x {height} pixels. Coordinate system origin is top-left,
x increases rightward, y increases downward. Return integer pixel coordinates only —
every x and y must be a whole number, never a float.

Ignore all text, labels, dimensions, measurement lines, scale bars, legends, furniture,
fixtures, and annotation graphics. Trace only the architectural floor plan geometry.

Room boundaries must coincide with visible wall lines in the floor plan. Each room polygon
is a simple closed boundary with at least 3 vertices, aligned to wall edges. Where a room
border crosses a door or window opening, continue the room boundary as a straight line
segment across the opening along the wall axis — do not notch inward around the opening.

Return:
- rooms: one polygon per separate white room region in the combined mask
- doors: one polygon per visible door opening (typically a thin rectangle across the wall gap)
- windows: one polygon per visible window opening (typically a thin rectangle across the wall gap)
- floorplan_boundary: one outer polygon enclosing the actual floor plan drawing area
  (the outermost built footprint / plan outline, excluding surrounding margin and annotations)"""


class Point(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x: int
    y: int


class PolygonResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = ""
    points: list[Point] = Field(min_length=3)
    note: str = ""


class FloorplanPolygonsTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rooms: list[PolygonResult] = Field(min_length=1)
    doors: list[PolygonResult] = Field(default_factory=list)
    windows: list[PolygonResult] = Field(default_factory=list)
    floorplan_boundary: PolygonResult
    limitations: list[str] = Field(default_factory=list)


def scale_image(image: Image.Image, size: int) -> Image.Image:
    if image.size == (size, size):
        return image.convert("RGB")
    return image.resize((size, size), Image.Resampling.LANCZOS).convert("RGB")


def scale_mask(mask: np.ndarray, size: int) -> np.ndarray:
    plane = mask > 0.5
    if plane.shape == (size, size):
        return plane
    return cv2.resize(
        plane.astype(np.uint8),
        (size, size),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def scale_polygon(polygon: PolygonRing, *, from_size: int, to_size: int) -> PolygonRing:
    if from_size == to_size:
        return [(int(x), int(y)) for x, y in polygon]
    scale = to_size / from_size
    return [(int(round(x * scale)), int(round(y * scale))) for x, y in polygon]


def scale_polygons(
    polygons: list[PolygonRing],
    *,
    from_size: int,
    to_size: int,
) -> list[PolygonRing]:
    return [scale_polygon(polygon, from_size=from_size, to_size=to_size) for polygon in polygons]


def _image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _usage_from_response(response: object) -> LlmTokenUsage:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {
            "input_tokens": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
    details = usage.input_tokens_details
    cached_tokens = int(details.cached_tokens) if details is not None else 0
    cache_write_tokens = int(details.cache_write_tokens) if details is not None else 0
    return {
        "input_tokens": int(usage.input_tokens),
        "cached_tokens": cached_tokens,
        "cache_write_tokens": cache_write_tokens,
        "output_tokens": int(usage.output_tokens),
        "total_tokens": int(usage.total_tokens),
    }


def llm_cost_usd(
    usage: LlmTokenUsage,
    *,
    input_cost_per_million_usd: float,
    cached_input_cost_per_million_usd: float,
    cache_write_cost_per_million_usd: float,
    output_cost_per_million_usd: float,
) -> float:
    cached_tokens = usage.get("cached_tokens", 0)
    cache_write_tokens = usage.get("cache_write_tokens", 0)
    billable_input = max(0, usage["input_tokens"] - cached_tokens - cache_write_tokens)
    return (
        billable_input * input_cost_per_million_usd
        + cached_tokens * cached_input_cost_per_million_usd
        + cache_write_tokens * cache_write_cost_per_million_usd
        + usage["output_tokens"] * output_cost_per_million_usd
    ) / 1_000_000


def _validate_polygon(points: list[list[int]], *, width: int, height: int) -> PolygonRing:
    polygon: PolygonRing = []
    for x, y in points:
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"Point ({x}, {y}) is outside {width}x{height}")
        polygon.append((int(x), int(y)))
    if len(polygon) < 3:
        raise ValueError("Polygon must have at least 3 vertices")
    return polygon


def _parse_polygon_entries(
    entries: list[PolygonResult],
    *,
    llm_size: int,
    label: str,
    log: StepLogFn | None = None,
) -> list[dict[str, object]]:
    refined: list[dict[str, object]] = []
    for index, entry in enumerate(entries):
        points = [[point.x, point.y] for point in entry.points]
        polygon = _validate_polygon(points, width=llm_size, height=llm_size)
        name = entry.name or f"{label}-{index:03d}"
        refined.append(
            {
                "index": index,
                "name": name,
                "polygon": polygon,
                "note": entry.note,
            }
        )
        if log is not None:
            log(f"llm {label} → {index:03d}, {name}, {len(polygon)} vertices")
    return refined


def _call_floorplan_polygons_llm(
    *,
    input_image_path: Path,
    room_mask_path: Path,
    llm_size: int,
    model: str,
    detail: str,
    api_key: str,
    log: StepLogFn | None = None,
) -> tuple[FloorplanPolygonsTrace, LlmTokenUsage]:
    if log is not None:
        log("llm call → floor plan + combined room mask")
    client = OpenAI(api_key=api_key)
    response = client.responses.parse(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            f"{SYSTEM_PROMPT.format(width=llm_size, height=llm_size)}\n\n"
                            "Image 1 is the floor plan. Image 2 is the combined rough room "
                            "mask (white = candidate room interiors)."
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": _image_data_url(input_image_path),
                        "detail": detail,
                    },
                    {
                        "type": "input_image",
                        "image_url": _image_data_url(room_mask_path),
                        "detail": detail,
                    },
                ],
            }
        ],
        text_format=FloorplanPolygonsTrace,
    )
    if response.output_parsed is None:
        raise RuntimeError("LLM returned no parsed floor plan polygons")

    usage = _usage_from_response(response)
    trace = response.output_parsed
    if log is not None:
        log(
            f"llm response → {len(trace.rooms)} rooms, "
            f"{len(trace.doors)} doors, {len(trace.windows)} windows, "
            f"tokens: {usage['total_tokens']:,}"
        )
    return trace, usage


def refine_room_polygons(
    input_image: Image.Image,
    room_mask: np.ndarray,
    out_dir: Path,
    *,
    llm_size: int,
    pipeline_size: int,
    min_area_px: int,
    model: str,
    detail: str,
    api_key: str,
    input_cost_per_million_usd: float,
    cached_input_cost_per_million_usd: float,
    cache_write_cost_per_million_usd: float,
    output_cost_per_million_usd: float,
    log: StepLogFn | None = None,
) -> tuple[
    list[PolygonRing],
    list[PolygonRing],
    list[PolygonRing],
    PolygonRing,
    dict[str, object],
]:
    out_dir.mkdir(parents=True, exist_ok=True)

    scaled_input = scale_image(input_image, llm_size)
    scaled_mask = scale_mask(room_mask, llm_size)
    scaled_min_area = max(1, round(min_area_px * (llm_size / pipeline_size) ** 2))

    input_path = out_dir / "input-1024.jpg"
    mask_path = out_dir / "room-mask-1024.png"
    scaled_input.save(input_path, format="JPEG", quality=95)
    Image.fromarray(scaled_mask.astype(np.uint8) * 255, mode="L").save(mask_path)

    mask_region_count = len(
        room_components_from_mask(scaled_mask, min_area_px=scaled_min_area)
    )
    if log is not None:
        log(
            f"room mask regions → {mask_region_count} "
            f"(min area {scaled_min_area}px)"
        )

    trace, usage = _call_floorplan_polygons_llm(
        input_image_path=input_path,
        room_mask_path=mask_path,
        llm_size=llm_size,
        model=model,
        detail=detail,
        api_key=api_key,
        log=log,
    )

    rooms = _parse_polygon_entries(trace.rooms, llm_size=llm_size, label="room", log=log)
    doors = _parse_polygon_entries(trace.doors, llm_size=llm_size, label="door", log=log)
    windows = _parse_polygon_entries(trace.windows, llm_size=llm_size, label="window", log=log)
    boundary_points = [[point.x, point.y] for point in trace.floorplan_boundary.points]
    boundary_llm = _validate_polygon(boundary_points, width=llm_size, height=llm_size)
    if log is not None:
        log(
            f"llm boundary → {trace.floorplan_boundary.name or 'floorplan'}, "
            f"{len(boundary_llm)} vertices"
        )

    room_polygons_llm = [entry["polygon"] for entry in rooms]
    door_polygons_llm = [entry["polygon"] for entry in doors]
    window_polygons_llm = [entry["polygon"] for entry in windows]

    room_polygons = scale_polygons(room_polygons_llm, from_size=llm_size, to_size=pipeline_size)
    door_polygons = scale_polygons(door_polygons_llm, from_size=llm_size, to_size=pipeline_size)
    window_polygons = scale_polygons(window_polygons_llm, from_size=llm_size, to_size=pipeline_size)
    floorplan_boundary = scale_polygon(boundary_llm, from_size=llm_size, to_size=pipeline_size)

    total_cost_usd = llm_cost_usd(
        usage,
        input_cost_per_million_usd=input_cost_per_million_usd,
        cached_input_cost_per_million_usd=cached_input_cost_per_million_usd,
        cache_write_cost_per_million_usd=cache_write_cost_per_million_usd,
        output_cost_per_million_usd=output_cost_per_million_usd,
    )
    usage_summary = {
        **usage,
        "cost_usd": total_cost_usd,
    }

    llm_payload = {
        "size": llm_size,
        "pipeline_size": pipeline_size,
        "model": model,
        "room_count": len(rooms),
        "door_count": len(doors),
        "window_count": len(windows),
        "usage": usage_summary,
        "limitations": trace.limitations,
        "rooms": [
            {
                "index": entry["index"],
                "name": entry["name"],
                "polygon": [[x, y] for x, y in entry["polygon"]],
                "note": entry.get("note", ""),
            }
            for entry in rooms
        ],
        "doors": [
            {
                "index": entry["index"],
                "name": entry["name"],
                "polygon": [[x, y] for x, y in entry["polygon"]],
                "note": entry.get("note", ""),
            }
            for entry in doors
        ],
        "windows": [
            {
                "index": entry["index"],
                "name": entry["name"],
                "polygon": [[x, y] for x, y in entry["polygon"]],
                "note": entry.get("note", ""),
            }
            for entry in windows
        ],
        "floorplan_boundary": {
            "name": trace.floorplan_boundary.name or "floorplan",
            "polygon": [[x, y] for x, y in boundary_llm],
            "note": trace.floorplan_boundary.note,
        },
    }
    llm_json_path = out_dir / "floorplan-1024.json"
    llm_json_path.write_text(json.dumps(llm_payload, indent=2), encoding="utf-8")

    meta = {
        "input_1024": input_path,
        "room_mask_1024": mask_path,
        "floorplan_1024_json": llm_json_path,
        "rooms": rooms,
        "doors": doors,
        "windows": windows,
        "floorplan_boundary": boundary_llm,
        "usage": usage_summary,
        "limitations": trace.limitations,
    }
    return room_polygons, door_polygons, window_polygons, floorplan_boundary, meta


def render_room_polygons(
    input_image: Image.Image,
    polygons: list[PolygonRing],
    *,
    size: int,
    png_path: Path,
    svg_path: Path,
    door_polygons: list[PolygonRing] | None = None,
    window_polygons: list[PolygonRing] | None = None,
    floorplan_boundary: PolygonRing | None = None,
) -> None:
    view = floorplan_view(input_image, size)
    overlay = view.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")
    outline = Image.new("RGB", (size, size), (251, 252, 254))
    outline_draw = ImageDraw.Draw(outline)

    colors = [
        (220, 50, 50),
        (50, 120, 220),
        (40, 170, 80),
        (230, 160, 40),
        (160, 60, 220),
    ]
    svg_paths: list[str] = []

    if floorplan_boundary and len(floorplan_boundary) >= 3:
        boundary_points = [(int(x), int(y)) for x, y in floorplan_boundary]
        draw.polygon(boundary_points, fill=(0, 0, 0, 20), outline=(20, 20, 20, 255))
        outline_draw.polygon(boundary_points, outline=(20, 20, 20), width=4)
        svg_paths.append(
            '    <polygon id="floorplan-boundary" '
            f'points="{" ".join(f"{x},{y}" for x, y in boundary_points)}" '
            'fill="none" stroke="#141414" stroke-width="3" />'
        )

    for index, polygon in enumerate(polygons):
        if len(polygon) < 3:
            continue
        color = colors[index % len(colors)]
        points = [(int(x), int(y)) for x, y in polygon]
        draw.polygon(points, fill=(*color, 80), outline=(*color, 255))
        outline_draw.polygon(points, outline=color, width=3)
        svg_paths.append(
            f'    <polygon id="room-{index:03d}" points="{" ".join(f"{x},{y}" for x, y in points)}" '
            f'fill="none" stroke="#173f6b" stroke-width="2" />'
        )

    for index, polygon in enumerate(door_polygons or []):
        if len(polygon) < 3:
            continue
        points = [(int(x), int(y)) for x, y in polygon]
        draw.polygon(points, fill=(40, 180, 80, 120), outline=(20, 140, 50, 255))
        outline_draw.polygon(points, outline=(20, 140, 50), width=2)
        svg_paths.append(
            f'    <polygon id="door-{index:03d}" points="{" ".join(f"{x},{y}" for x, y in points)}" '
            f'fill="none" stroke="#148c32" stroke-width="2" />'
        )

    for index, polygon in enumerate(window_polygons or []):
        if len(polygon) < 3:
            continue
        points = [(int(x), int(y)) for x, y in polygon]
        draw.polygon(points, fill=(60, 120, 240, 120), outline=(30, 80, 200, 255))
        outline_draw.polygon(points, outline=(30, 80, 200), width=2)
        svg_paths.append(
            f'    <polygon id="window-{index:03d}" points="{" ".join(f"{x},{y}" for x, y in points)}" '
            f'fill="none" stroke="#1e50c8" stroke-width="2" />'
        )

    panel = merge_panels(
        [
            view,
            with_label(overlay.convert("RGB"), "Polygons on input"),
            with_label(outline, "Polygons only"),
        ]
    )
    png_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(png_path, format="JPEG", quality=95)

    svg = (
        f'<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}">\n'
        f'  <g fill="none" stroke="#173f6b" stroke-width="2">\n'
        + "\n".join(svg_paths)
        + "\n  </g>\n</svg>\n"
    )
    svg_path.write_text(svg, encoding="utf-8")


_PREVIEW_ALPHA = 120


def _room_preview_color(index: int) -> tuple[int, int, int]:
    hue = (index * 0.61803398875) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.62, 0.92)
    return int(red * 255), int(green * 255), int(blue * 255)


def render_final_mask_polygon_overlay(
    input_image: Image.Image,
    *,
    room_polygons: list[PolygonRing],
    wall_polygons: list[PolygonRing] | None = None,
    door_polygons: list[PolygonRing],
    window_polygons: list[PolygonRing],
) -> Image.Image:
    """Return input with translucent room/wall/door/window polygon overlays."""
    base = input_image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    for index, polygon in enumerate(room_polygons):
        if len(polygon) < 3:
            continue
        rgb = _room_preview_color(index)
        draw.polygon(polygon, fill=(*rgb, _PREVIEW_ALPHA), outline=(*rgb, 220))
    for polygon in wall_polygons or []:
        if len(polygon) >= 3:
            draw.polygon(polygon, fill=(0, 100, 0, _PREVIEW_ALPHA), outline=(0, 80, 0, 220))
    for polygon in door_polygons:
        if len(polygon) >= 3:
            draw.polygon(polygon, fill=(80, 220, 80, _PREVIEW_ALPHA), outline=(40, 180, 40, 220))
    for polygon in window_polygons:
        if len(polygon) >= 3:
            draw.polygon(polygon, fill=(80, 160, 255, _PREVIEW_ALPHA), outline=(40, 120, 220, 220))
    return Image.alpha_composite(base, overlay).convert("RGB")


def save_final_mask_polygon_preview(
    input_image: Image.Image,
    *,
    room_polygons: list[PolygonRing],
    wall_polygons: list[PolygonRing] | None = None,
    door_polygons: list[PolygonRing],
    window_polygons: list[PolygonRing],
    out_dir: Path,
    save_input: bool = True,
) -> tuple[Path | None, Path]:
    """Write a translucent per-polygon ``polygon-overlay.jpg``; optionally ``input.jpg``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    input_path: Path | None = None
    if save_input:
        input_path = out_dir / "input.jpg"
        input_image.convert("RGB").save(input_path, format="JPEG", quality=95)

    overlay_path = out_dir / "polygon-overlay.jpg"
    render_final_mask_polygon_overlay(
        input_image,
        room_polygons=room_polygons,
        wall_polygons=wall_polygons,
        door_polygons=door_polygons,
        window_polygons=window_polygons,
    ).save(overlay_path, format="JPEG", quality=95)
    return input_path, overlay_path


def save_room_polygon_outputs(
    step_dir: Path,
    *,
    pipeline_size: int,
    min_area_px: int,
    room_polygons: list[PolygonRing],
    door_polygons: list[PolygonRing],
    window_polygons: list[PolygonRing],
    floorplan_boundary: PolygonRing,
    wall_approx_epsilon: float,
    input_image: Image.Image | None = None,
) -> dict[str, Path | dict[str, Path]]:
    final_dir = step_dir / "final-masks"
    polygon_paths = save_final_mask_contour_polygons(
        final_dir,
        size=pipeline_size,
        min_area_px=min_area_px,
        room_polygons=room_polygons,
        door_polygons=door_polygons,
        window_polygons=window_polygons,
        boundary_polygon=floorplan_boundary,
        wall_approx_epsilon=wall_approx_epsilon,
    )
    paths: dict[str, Path | dict[str, Path]] = {
        "background": final_dir / "background.png",
        "room": final_dir / "room.png",
        "wall": final_dir / "wall.png",
        "door": final_dir / "door.png",
        "window": final_dir / "window.png",
        "polygons": polygon_paths,
    }
    if input_image is not None:
        _, overlay_path = save_final_mask_polygon_preview(
            input_image,
            room_polygons=room_polygons,
            wall_polygons=load_layer_polygons(polygon_paths["wall"]),
            door_polygons=door_polygons,
            window_polygons=window_polygons,
            out_dir=final_dir,
            save_input=True,
        )
        paths["input"] = final_dir / "input.jpg"
        paths["polygon_overlay"] = overlay_path
    return paths
