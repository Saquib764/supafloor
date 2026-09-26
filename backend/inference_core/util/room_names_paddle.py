"""PaddleOCR room name + label-position detection from a floor plan image."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from util.room_polygon_llm import scale_image

StepLogFn = Callable[[str], None]

PADDLE_MODEL_NAME = "PP-OCRv5_mobile"
_PURE_NUMBER_RE = re.compile(r"^\d+([.,]\d+)?$")
_DIMENSION_RE = re.compile(
    r"^\d+([.,]\d+)?\s*('|\"|m|cm|mm|ft|in|sq\.?ft|sqft|°)\.?$",
    re.IGNORECASE,
)
_AREA_RE = re.compile(
    r"^\d+([.,]\d+)?\s*(m2|m²|sqm|sq\.?\s*m|sq\.?ft|sqft)\.?$",
    re.IGNORECASE,
)

_OCR_CACHE: dict[tuple[str, bool, int], Any] = {}


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


def _box_center(box: list[list[float]]) -> dict[str, int]:
    xs = [point[0] for point in box]
    ys = [point[1] for point in box]
    return {
        "x": int(round(sum(xs) / len(xs))),
        "y": int(round(sum(ys) / len(ys))),
    }


def _is_room_label(text: str, *, filter_dimensions: bool) -> bool:
    cleaned = text.strip().lower()
    if len(cleaned) <= 1:
        return False
    if not any(char.isalpha() for char in cleaned):
        return False
    if not filter_dimensions:
        return True
    if _PURE_NUMBER_RE.fullmatch(cleaned):
        return False
    if _DIMENSION_RE.fullmatch(cleaned):
        return False
    if _AREA_RE.fullmatch(cleaned.replace(" ", "")):
        return False
    if _AREA_RE.fullmatch(cleaned):
        return False
    return True


def _center_in_mask(x: int, y: int, room_mask: np.ndarray) -> bool:
    height, width = room_mask.shape
    if x < 0 or y < 0 or x >= width or y >= height:
        return False
    return bool(room_mask[y, x])


def _get_paddle_ocr(*, lang: str, use_angle_cls: bool, det_limit_side_len: int) -> Any:
    cache_key = (lang, use_angle_cls, det_limit_side_len)
    cached = _OCR_CACHE.get(cache_key)
    if cached is not None:
        return cached
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise RuntimeError(
            "PaddleOCR is not installed. Run `uv sync` in backend/inference_core."
        ) from exc

    ocr = PaddleOCR(
        lang=lang,
        use_angle_cls=use_angle_cls,
        det_limit_side_len=det_limit_side_len,
        show_log=False,
    )
    _OCR_CACHE[cache_key] = ocr
    return ocr


def _run_paddle_ocr(
    image: Image.Image,
    *,
    lang: str,
    use_angle_cls: bool,
    det_limit_side_len: int,
    log: StepLogFn | None = None,
) -> list[dict[str, object]]:
    if log is not None:
        log("paddle ocr → detect + recognize room labels")
    ocr = _get_paddle_ocr(
        lang=lang,
        use_angle_cls=use_angle_cls,
        det_limit_side_len=det_limit_side_len,
    )
    image_array = np.array(image.convert("RGB"))
    raw_result = ocr.ocr(image_array, cls=use_angle_cls)
    if not raw_result or raw_result[0] is None:
        return []

    detections: list[dict[str, object]] = []
    for line in raw_result[0]:
        box, text_info = line
        text, confidence = text_info
        normalized = str(text).strip().lower()
        if not normalized:
            continue
        detections.append(
            {
                "room": normalized,
                "position": _box_center(box),
                "confidence": float(confidence),
                "box": [[int(point[0]), int(point[1])] for point in box],
            }
        )
    return detections


def detect_room_names(
    input_image: Image.Image,
    out_dir: Path,
    *,
    ocr_size: int,
    pipeline_size: int,
    lang: str,
    use_angle_cls: bool,
    det_limit_side_len: int,
    min_confidence: float,
    filter_dimensions: bool,
    require_inside_room_mask: bool,
    room_mask: np.ndarray | None = None,
    log: StepLogFn | None = None,
) -> dict[str, object]:
    """Detect room label names and positions; write ``room-names.json``."""
    out_dir.mkdir(parents=True, exist_ok=True)

    scaled_input = scale_image(input_image, ocr_size)
    input_path = out_dir / f"input-{ocr_size}.jpg"
    scaled_input.save(input_path, format="JPEG", quality=95)

    started = time.perf_counter()
    raw_detections = _run_paddle_ocr(
        scaled_input,
        lang=lang,
        use_angle_cls=use_angle_cls,
        det_limit_side_len=det_limit_side_len,
        log=log,
    )
    elapsed_ms = int(round((time.perf_counter() - started) * 1000))

    rooms_ocr: list[dict[str, object]] = []
    for detection in raw_detections:
        confidence = float(detection["confidence"])
        if confidence < min_confidence:
            continue
        text = str(detection["room"])
        if not _is_room_label(text, filter_dimensions=filter_dimensions):
            continue
        position = detection["position"]
        assert isinstance(position, dict)
        if require_inside_room_mask:
            if room_mask is None:
                raise RuntimeError(
                    "require_inside_room_mask is enabled but no room mask was provided"
                )
            if not _center_in_mask(int(position["x"]), int(position["y"]), room_mask):
                continue
        rooms_ocr.append(
            {
                "room": text,
                "position": {"x": int(position["x"]), "y": int(position["y"])},
                "confidence": confidence,
                "box": detection["box"],
            }
        )

    rooms = [
        {
            "room": entry["room"],
            "position": _scale_position(
                entry["position"],  # type: ignore[arg-type]
                from_size=ocr_size,
                to_size=pipeline_size,
            ),
        }
        for entry in rooms_ocr
    ]

    for index, entry in enumerate(rooms):
        pos = entry["position"]
        if log is not None:
            log(f"paddle room → {index:03d}, {entry['room']}, ({pos['x']}, {pos['y']})")

    if log is not None:
        log(
            f"paddle response → {len(rooms)} room labels kept from "
            f"{len(raw_detections)} detections in {elapsed_ms}ms"
        )

    usage_summary = {
        "detections": len(raw_detections),
        "kept": len(rooms),
        "elapsed_ms": elapsed_ms,
    }

    payload = {
        "backend": "paddle",
        "width": pipeline_size,
        "height": pipeline_size,
        "ocr_size": ocr_size,
        "model": PADDLE_MODEL_NAME,
        "rooms": rooms,
        "rooms_raw": rooms_ocr,
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
