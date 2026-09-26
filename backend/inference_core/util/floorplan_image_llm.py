"""LLM full floor plan extraction from a single image at 1024×1024."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

from openai import OpenAI
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from util.canvas import fit_to_canvas_rgb
from util.final_mask_polygons import PolygonRing
from util.room_polygon_llm import (
    _image_data_url,
    _parse_polygon_entries,
    _usage_from_response,
    _validate_polygon,
    llm_cost_usd,
)

StepLogFn = Callable[[str], None]

SYSTEM_PROMPT = """You are a precise floor plan tracing and room-label reading system.
Image size is exactly {width} x {height} pixels. Coordinate system origin is top-left,
x increases rightward, y increases downward. Return integer pixel coordinates only —
every x and y must be a whole number in [0, {max_coord}].

Trace the architectural floor plan geometry from the image:
- rooms: one closed polygon per distinct room interior, aligned to visible wall lines
- doors: one polygon per visible door opening (typically a thin rectangle across the wall gap)
- windows: one polygon per visible window opening (typically a thin rectangle across the wall gap)
- boundary: one outer polygon enclosing the built floor plan footprint (exclude margins and annotations)

Where a room border crosses a door or window opening, continue the room boundary as a straight
line segment across the opening along the wall axis — do not notch inward around the opening.

Read every visible room name or label printed on the floor plan:
- names: one entry per visible text label with room (lowercase label text) and position (label center)

Ignore dimensions, scale bars, legends, furniture, fixtures, and annotation graphics when tracing walls.
Only include names entries for rooms that have visible text labels on the plan."""


class Point(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x: int
    y: int


class PolygonEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = ""
    points: list[Point] = Field(min_length=3)
    note: str = ""


class NameEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    room: str = Field(min_length=1)
    position: Point


class FloorplanImageExtract(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rooms: list[PolygonEntry] = Field(min_length=1)
    doors: list[PolygonEntry] = Field(default_factory=list)
    windows: list[PolygonEntry] = Field(default_factory=list)
    boundary: list[Point] = Field(min_length=3)
    names: list[NameEntry] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


def _call_floorplan_image_llm(
    *,
    input_image_path: Path,
    llm_size: int,
    model: str,
    detail: str,
    api_key: str,
    log: StepLogFn | None = None,
) -> tuple[FloorplanImageExtract, dict[str, int]]:
    if log is not None:
        log(f"llm call → floor plan image extract ({model} at {llm_size}px)")
    client = OpenAI(api_key=api_key)
    max_coord = llm_size - 1
    response = client.responses.parse(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            f"{SYSTEM_PROMPT.format(width=llm_size, height=llm_size, max_coord=max_coord)}\n\n"
                            "The image is a floor plan. Return rooms, doors, windows, boundary, and names."
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": _image_data_url(input_image_path),
                        "detail": detail,
                    },
                ],
            }
        ],
        text_format=FloorplanImageExtract,
    )
    if response.output_parsed is None:
        raise RuntimeError("LLM returned no parsed floor plan extract")

    usage = _usage_from_response(response)
    trace = response.output_parsed
    if log is not None:
        log(
            f"llm response → {len(trace.rooms)} rooms, "
            f"{len(trace.doors)} doors, {len(trace.windows)} windows, "
            f"{len(trace.names)} names, tokens: {usage['total_tokens']:,}"
        )
    return trace, usage


def extract_floorplan_from_image(
    input_image: Image.Image,
    out_dir: Path,
    *,
    llm_size: int,
    model: str,
    detail: str,
    api_key: str,
    input_cost_per_million_usd: float,
    cached_input_cost_per_million_usd: float,
    cache_write_cost_per_million_usd: float,
    output_cost_per_million_usd: float,
    log: StepLogFn | None = None,
) -> dict[str, object]:
    """Extract rooms, doors, windows, boundary, and names; write ``floorplan.json``."""
    out_dir.mkdir(parents=True, exist_ok=True)

    scaled_input = fit_to_canvas_rgb(input_image, size=llm_size)
    input_path = out_dir / "input-1024.jpg"
    scaled_input.save(input_path, format="JPEG", quality=95)

    started = time.perf_counter()
    trace, usage = _call_floorplan_image_llm(
        input_image_path=input_path,
        llm_size=llm_size,
        model=model,
        detail=detail,
        api_key=api_key,
        log=log,
    )
    elapsed_seconds = time.perf_counter() - started

    rooms_parsed = _parse_polygon_entries(trace.rooms, llm_size=llm_size, label="room", log=log)
    doors_parsed = _parse_polygon_entries(trace.doors, llm_size=llm_size, label="door", log=log)
    windows_parsed = _parse_polygon_entries(trace.windows, llm_size=llm_size, label="window", log=log)
    boundary_points = [[point.x, point.y] for point in trace.boundary]
    boundary = _validate_polygon(boundary_points, width=llm_size, height=llm_size)
    if log is not None:
        log(f"llm boundary → {len(boundary)} vertices")

    rooms = [entry["polygon"] for entry in rooms_parsed]
    doors = [entry["polygon"] for entry in doors_parsed]
    windows = [entry["polygon"] for entry in windows_parsed]
    names = [
        {
            "room": entry.room.strip().lower(),
            "position": {"x": entry.position.x, "y": entry.position.y},
        }
        for entry in trace.names
    ]

    total_cost_usd = llm_cost_usd(
        usage,
        input_cost_per_million_usd=input_cost_per_million_usd,
        cached_input_cost_per_million_usd=cached_input_cost_per_million_usd,
        cache_write_cost_per_million_usd=cache_write_cost_per_million_usd,
        output_cost_per_million_usd=output_cost_per_million_usd,
    )
    usage_summary = {**usage, "cost_usd": total_cost_usd}
    if log is not None:
        log(
            f"llm extract done → {model}, "
            f"tokens: {usage['input_tokens']:,} in + {usage['output_tokens']:,} out, "
            f"cost ${total_cost_usd:.4f}, "
            f"time {elapsed_seconds:.1f}s"
        )

    payload = {
        "size": llm_size,
        "model": model,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "usage": usage_summary,
        "limitations": trace.limitations,
        "rooms": [[[x, y] for x, y in polygon] for polygon in rooms],
        "doors": [[[x, y] for x, y in polygon] for polygon in doors],
        "windows": [[[x, y] for x, y in polygon] for polygon in windows],
        "boundary": [[x, y] for x, y in boundary],
        "names": names,
    }
    json_path = out_dir / "floorplan.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return {
        "input": input_path,
        "json": json_path,
        "size": llm_size,
        "rooms": rooms,
        "doors": doors,
        "windows": windows,
        "boundary": boundary,
        "names": names,
        "usage": usage_summary,
        "elapsed_seconds": elapsed_seconds,
    }


def load_floorplan_extract(json_path: Path) -> dict[str, object] | None:
    if not json_path.is_file():
        return None
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None

    def _load_polygons(key: str) -> list[PolygonRing]:
        raw = payload.get(key)
        if not isinstance(raw, list):
            return []
        polygons: list[PolygonRing] = []
        for entry in raw:
            if not isinstance(entry, list):
                continue
            ring: PolygonRing = []
            for point in entry:
                if isinstance(point, list) and len(point) == 2:
                    ring.append((int(point[0]), int(point[1])))
            if len(ring) >= 3:
                polygons.append(ring)
        return polygons

    boundary_raw = payload.get("boundary")
    boundary: PolygonRing = []
    if isinstance(boundary_raw, list):
        for point in boundary_raw:
            if isinstance(point, list) and len(point) == 2:
                boundary.append((int(point[0]), int(point[1])))

    names_raw = payload.get("names")
    names: list[dict[str, object]] = []
    if isinstance(names_raw, list):
        for entry in names_raw:
            if not isinstance(entry, dict):
                continue
            position = entry.get("position")
            if not isinstance(position, dict):
                continue
            names.append(
                {
                    "room": str(entry.get("room") or ""),
                    "position": {
                        "x": int(position.get("x", 0)),
                        "y": int(position.get("y", 0)),
                    },
                }
            )

    usage = payload.get("usage")
    elapsed_raw = payload.get("elapsed_seconds")
    elapsed_seconds = float(elapsed_raw) if elapsed_raw is not None else None
    return {
        "size": int(payload.get("size", 1024)),
        "model": str(payload.get("model", "")),
        "elapsed_seconds": elapsed_seconds,
        "rooms": _load_polygons("rooms"),
        "doors": _load_polygons("doors"),
        "windows": _load_polygons("windows"),
        "boundary": boundary if len(boundary) >= 3 else [],
        "names": names,
        "usage": usage if isinstance(usage, dict) else {},
    }
