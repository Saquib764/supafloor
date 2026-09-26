"""LLM room name + label-position detection from a floor plan image."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from openai import OpenAI
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from util.room_polygon_llm import (
    _image_data_url,
    _usage_from_response,
    llm_cost_usd,
    scale_image,
)

StepLogFn = Callable[[str], None]

SYSTEM_PROMPT = """You are a precise floor plan room-label reader.
Image size is exactly {width} x {height} pixels. Coordinate system origin is top-left,
x increases rightward, y increases downward. Return integer pixel coordinates only.

Read every visible room name or label printed on the floor plan drawing.
For each labeled room, return:
- room: the label text normalized to lowercase (e.g. "kitchen", "bedroom 1", "bath")
- position: the pixel center of that label text on the image

Ignore dimensions, scale bars, legends, furniture labels, and annotation graphics.
Only include rooms that have a visible text label on the plan."""


class Point(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x: int
    y: int


class RoomNameEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    room: str = Field(min_length=1)
    position: Point


class RoomNamesTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rooms: list[RoomNameEntry] = Field(default_factory=list)


def _scale_position(
    position: dict[str, int],
    *,
    from_size: int,
    to_size: int,
) -> dict[str, int]:
    if from_size == to_size:
        return {"x": int(position["x"]), "y": int(position["y"])}
    scale = to_size / from_size
    return {
        "x": int(round(position["x"] * scale)),
        "y": int(round(position["y"] * scale)),
    }


def _call_room_names_llm(
    *,
    input_image_path: Path,
    llm_size: int,
    model: str,
    detail: str,
    api_key: str,
    log: StepLogFn | None = None,
) -> tuple[RoomNamesTrace, dict[str, int]]:
    if log is not None:
        log("llm call → room names + label positions")
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
                            "The image is a floor plan. Return one entry per visible room label."
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
        text_format=RoomNamesTrace,
    )
    if response.output_parsed is None:
        raise RuntimeError("LLM returned no parsed room names")

    usage = _usage_from_response(response)
    trace = response.output_parsed
    if log is not None:
        log(
            f"llm response → {len(trace.rooms)} room labels, "
            f"tokens: {usage['total_tokens']:,}"
        )
    return trace, usage


def detect_room_names(
    input_image: Image.Image,
    out_dir: Path,
    *,
    llm_size: int,
    pipeline_size: int,
    model: str,
    detail: str,
    api_key: str,
    input_cost_per_million_usd: float,
    cached_input_cost_per_million_usd: float,
    cache_write_cost_per_million_usd: float,
    output_cost_per_million_usd: float,
    log: StepLogFn | None = None,
) -> dict[str, object]:
    """Detect room label names and positions; write ``room-names.json``."""
    out_dir.mkdir(parents=True, exist_ok=True)

    scaled_input = scale_image(input_image, llm_size)
    input_path = out_dir / f"input-{llm_size}.jpg"
    scaled_input.save(input_path, format="JPEG", quality=95)

    trace, usage = _call_room_names_llm(
        input_image_path=input_path,
        llm_size=llm_size,
        model=model,
        detail=detail,
        api_key=api_key,
        log=log,
    )

    rooms_llm = [
        {
            "room": entry.room.strip().lower(),
            "position": {"x": entry.position.x, "y": entry.position.y},
        }
        for entry in trace.rooms
    ]
    rooms = [
        {
            "room": entry["room"],
            "position": _scale_position(
                entry["position"],
                from_size=llm_size,
                to_size=pipeline_size,
            ),
        }
        for entry in rooms_llm
    ]

    for index, entry in enumerate(rooms):
        pos = entry["position"]
        if log is not None:
            log(f"llm room → {index:03d}, {entry['room']}, ({pos['x']}, {pos['y']})")

    total_cost_usd = llm_cost_usd(
        usage,
        input_cost_per_million_usd=input_cost_per_million_usd,
        cached_input_cost_per_million_usd=cached_input_cost_per_million_usd,
        cache_write_cost_per_million_usd=cache_write_cost_per_million_usd,
        output_cost_per_million_usd=output_cost_per_million_usd,
    )
    usage_summary = {**usage, "cost_usd": total_cost_usd}

    payload = {
        "backend": "llm",
        "width": pipeline_size,
        "height": pipeline_size,
        "ocr_size": llm_size,
        "llm_size": llm_size,
        "model": model,
        "rooms": rooms,
        "rooms_raw": rooms_llm,
        "rooms_llm": rooms_llm,
        "usage": usage_summary,
    }
    json_path = out_dir / "room-names.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return {
        "input": input_path,
        "json": json_path,
        "rooms": rooms,
        "usage": usage_summary,
    }
