"""Orchestrate vision LLM extraction of room polygons from floor plan images."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from extraction.constants import CANVAS_SIZE, EXTRACTION_MODEL, LLM_PROVIDER, REPLICATE_MODEL
from extraction.llm import image_part, parse_structured_messages
from extraction.plot import plot_rooms
from extraction.preprocess import prepare_canvas
from extraction.prompts import render_prompt
from extraction.schemas import ExtractionResponse, extraction_response_model

_DEFAULT_PROMPT = "v1"


def _prompt_cache_key(canvas_png: bytes, prompt: str) -> str:
    digest = hashlib.sha256(canvas_png).hexdigest()[:16]
    return f"extraction:{prompt}:{digest}"


def _output_stem(image: bytes | Path | str, stem: str | None) -> str:
    if stem is not None:
        return stem
    if isinstance(image, (str, Path)):
        return Path(image).stem
    return "floorplan"


def _make_meta(
    *,
    canvas_size: int,
    prompt: str,
    used_model: str,
    preprocess_meta: dict[str, Any],
    rooms: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "canvas_size": canvas_size,
        "model": used_model,
        "prompt": prompt,
        "preprocess": preprocess_meta,
        "room_count": len(rooms),
    }


def _save_outputs(
    out_dir: Path,
    stem: str,
    rooms: list[dict[str, Any]],
    meta: dict[str, Any],
    *,
    plot_png: bytes,
    debug_png: bytes,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}.rooms.json").write_text(json.dumps({"rooms": rooms, "meta": meta}, indent=2))
    (out_dir / f"{stem}_out.png").write_bytes(plot_png)
    (out_dir / f"{stem}_debug.png").write_bytes(debug_png)


def _normalize_rooms(parsed: ExtractionResponse) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for room in parsed.rooms:
        seen: set[tuple[int, int]] = set()
        polygon: list[dict[str, int]] = []
        for pt in room.polygon:
            key = (pt.x, pt.y)
            if key in seen:
                continue
            seen.add(key)
            polygon.append({"x": pt.x, "y": pt.y})
        if len(polygon) >= 3:
            out.append({"name": room.name.strip(), "polygon": polygon})
    return out


def extract_rooms(
    image: bytes | Path | str,
    *,
    canvas_size: int = CANVAS_SIZE,
    prompt: str = _DEFAULT_PROMPT,
    out_dir: Path | str | None = None,
    stem: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], bytes, bytes]:
    """Extract room polygons from a floor plan image.

    Returns:
        rooms — room polygons
        meta — execution metadata
        plot_png — polygon plot PNG
        debug_png — side-by-side canvas + polygon PNG
    """
    canvas_png, canvas_img, preprocess_meta = prepare_canvas(image, canvas_size=canvas_size)
    system, user = render_prompt(prompt, canvas_size=canvas_size)
    used_model = REPLICATE_MODEL if LLM_PROVIDER == "replicate" else EXTRACTION_MODEL
    save_dir = Path(out_dir) if out_dir is not None else None
    file_stem = _output_stem(image, stem)
    cache_key = _prompt_cache_key(canvas_png, prompt)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user},
                image_part(canvas_png),
            ],
        },
    ]

    response_model = (
        ExtractionResponse if canvas_size == CANVAS_SIZE else extraction_response_model(canvas_size)
    )
    parsed = parse_structured_messages(
        messages,
        response_model,
        model=used_model,
        prompt_cache_key=cache_key,
        image_png=canvas_png,
    )
    rooms = _normalize_rooms(parsed)
    plot_png = plot_rooms(rooms, canvas_size=canvas_size)
    debug_png = plot_rooms(rooms, background=canvas_img, canvas_size=canvas_size)
    meta = _make_meta(
        canvas_size=canvas_size,
        prompt=prompt,
        used_model=used_model,
        preprocess_meta=preprocess_meta,
        rooms=rooms,
    )
    if save_dir is not None:
        _save_outputs(save_dir, file_stem, rooms, meta, plot_png=plot_png, debug_png=debug_png)
    return rooms, meta, plot_png, debug_png
