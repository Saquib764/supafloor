"""Polygon exports for final FPN structure masks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from util.canvas import fit_to_canvas_placement
from util.mask_clean import morph_erode_dilate
from util.mask_fit import fit_mask_variants, pick_best_variant
from util.room_components import component_masks_from_binary

PolygonRing = list[tuple[int, int]]
OpeningLayer = Literal["door", "window"]
StructureLayer = Literal["room", "wall", "door", "window"]


def _component_contour(mask: np.ndarray) -> np.ndarray | None:
    binary = np.asarray(mask, dtype=bool).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _fit_room_polygon(
    mask: np.ndarray,
    *,
    width: int,
    height: int,
    min_sides: int,
    max_sides: int,
) -> dict[str, Any] | None:
    fit_mask = morph_erode_dilate(mask, 3)
    if not fit_mask.any():
        return None
    variants = fit_mask_variants(
        fit_mask,
        width=width,
        height=height,
        min_sides=min_sides,
        max_sides=max_sides,
    )
    best = pick_best_variant(variants)
    if best is None:
        return None
    return {
        "sides": best.sides,
        "score": best.score,
        "mask_minus_poly": best.mask_minus_poly,
        "poly_minus_mask": best.poly_minus_mask,
        "polygon": best.polygon,
    }


def _fit_wall_polygon(
    mask: np.ndarray,
    *,
    width: int,
    height: int,
    min_sides: int,
    max_sides: int,
) -> dict[str, Any] | None:
    """Fit wall component contour to the lowest-loss N-gon (4–15 sides)."""
    fit_mask = morph_erode_dilate(mask, 3)
    if not fit_mask.any():
        return None
    variants = fit_mask_variants(
        fit_mask,
        width=width,
        height=height,
        min_sides=min_sides,
        max_sides=max_sides,
    )
    best = pick_best_variant(variants)
    if best is None:
        return None
    return {
        "sides": best.sides,
        "score": best.score,
        "mask_minus_poly": best.mask_minus_poly,
        "poly_minus_mask": best.poly_minus_mask,
        "polygon": best.polygon,
    }


def _fit_opening_rectangle(mask: np.ndarray) -> dict[str, Any] | None:
    contour = _component_contour(mask)
    if contour is None or len(contour) < 3:
        return None
    rect = cv2.minAreaRect(contour)
    center_x, center_y = rect[0]
    rect_w, rect_h = rect[1]
    angle_deg = rect[2]
    corners = [
        [int(round(x)), int(round(y))]
        for x, y in cv2.boxPoints(rect)
    ]
    return {
        "center": [round(center_x, 2), round(center_y, 2)],
        "width": round(float(rect_w), 2),
        "height": round(float(rect_h), 2),
        "angle_deg": round(float(angle_deg), 2),
        "corners": corners,
    }


def _filtered_components(mask: np.ndarray, *, min_area_px: int) -> list[np.ndarray]:
    return [
        component
        for component in component_masks_from_binary(mask.astype(np.uint8))
        if int(component.sum()) >= min_area_px
    ]


def _write_layer_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _entry_polygon(entry: dict[str, Any]) -> PolygonRing | None:
    raw = entry.get("polygon") or entry.get("corners")
    if not raw or len(raw) < 3:
        return None
    return [(int(point[0]), int(point[1])) for point in raw]


def _step_canvas_point_to_full(
    x: int,
    y: int,
    *,
    crop_box: tuple[int, int, int, int],
    step_size: int,
    content_size: tuple[int, int],
) -> tuple[int, int]:
    x0, y0, _, _ = crop_box
    content_w, content_h = content_size
    paste_x, paste_y, fitted_w, fitted_h = fit_to_canvas_placement(
        content_w,
        content_h,
        size=step_size,
    )
    rel_x = (x - paste_x) * content_w / fitted_w
    rel_y = (y - paste_y) * content_h / fitted_h
    return int(round(x0 + rel_x)), int(round(y0 + rel_y))


def transform_polygons_to_full_canvas(
    components: list[PolygonRing],
    *,
    crop_box: tuple[int, int, int, int],
    step_size: int,
    content_size: tuple[int, int],
) -> list[PolygonRing]:
    return [
        [
            _step_canvas_point_to_full(
                x,
                y,
                crop_box=crop_box,
                step_size=step_size,
                content_size=content_size,
            )
            for x, y in ring
        ]
        for ring in components
        if len(ring) >= 3
    ]


def transform_mask_to_full_canvas(
    mask: np.ndarray,
    *,
    full_size: int,
    crop_box: tuple[int, int, int, int],
    step_size: int,
    content_size: tuple[int, int],
) -> np.ndarray:
    x0, y0, x1, y1 = crop_box
    content_w, content_h = content_size
    crop_w = max(1, x1 - x0)
    crop_h = max(1, y1 - y0)
    plane = mask > 0.5
    if plane.shape != (step_size, step_size):
        plane = cv2.resize(
            plane.astype(np.uint8),
            (step_size, step_size),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    paste_x, paste_y, fitted_w, fitted_h = fit_to_canvas_placement(
        content_w,
        content_h,
        size=step_size,
    )
    content = plane[paste_y : paste_y + fitted_h, paste_x : paste_x + fitted_w]
    resized = cv2.resize(
        content.astype(np.uint8),
        (crop_w, crop_h),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    full = np.zeros((full_size, full_size), dtype=bool)
    full[y0 : y0 + crop_h, x0 : x0 + crop_w] = resized
    return full


def _full_canvas_point_to_step(
    x: int,
    y: int,
    *,
    crop_box: tuple[int, int, int, int],
    step_size: int,
    content_size: tuple[int, int],
) -> tuple[int, int]:
    x0, y0, _, _ = crop_box
    content_w, content_h = content_size
    paste_x, paste_y, fitted_w, fitted_h = fit_to_canvas_placement(
        content_w,
        content_h,
        size=step_size,
    )
    rel_x = x - x0
    rel_y = y - y0
    step_x = paste_x + rel_x * fitted_w / content_w
    step_y = paste_y + rel_y * fitted_h / content_h
    return int(round(step_x)), int(round(step_y))


def transform_polygons_from_full_canvas(
    components: list[PolygonRing],
    *,
    crop_box: tuple[int, int, int, int],
    step_size: int,
    content_size: tuple[int, int],
) -> list[PolygonRing]:
    return [
        [
            _full_canvas_point_to_step(
                x,
                y,
                crop_box=crop_box,
                step_size=step_size,
                content_size=content_size,
            )
            for x, y in ring
        ]
        for ring in components
        if len(ring) >= 3
    ]


def load_layer_polygons(path: Path) -> list[PolygonRing]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    components: list[PolygonRing] = []
    for entry in data.get("components", []):
        polygon = _entry_polygon(entry)
        if polygon is not None:
            components.append(polygon)
    return components


def polygons_to_mask(components: list[PolygonRing], size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    for polygon in components:
        if len(polygon) < 3:
            continue
        fill = np.zeros((size, size), dtype=np.uint8)
        cv2.fillPoly(fill, [np.array(polygon, dtype=np.int32)], 1)
        mask |= fill.astype(bool)
    return mask


def _opening_entry(index: int, polygon: PolygonRing) -> dict[str, Any]:
    contour = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
    rect = cv2.minAreaRect(contour)
    center_x, center_y = rect[0]
    rect_w, rect_h = rect[1]
    corners = [
        [int(round(x)), int(round(y))]
        for x, y in cv2.boxPoints(rect)
    ]
    return {
        "index": index,
        "area_px": int(max(cv2.contourArea(contour), 1.0)),
        "center": [round(center_x, 2), round(center_y, 2)],
        "width": round(float(rect_w), 2),
        "height": round(float(rect_h), 2),
        "angle_deg": round(float(rect[2]), 2),
        "corners": corners,
    }


def save_layer_polygons(
    path: Path,
    layer: StructureLayer,
    components: list[PolygonRing],
    *,
    size: int,
    min_area_px: int = 0,
) -> Path:
    entries: list[dict[str, Any]] = []
    for index, polygon in enumerate(components):
        if len(polygon) < 3:
            continue
        if layer in ("door", "window"):
            entries.append(_opening_entry(index, polygon))
            continue
        fill = np.zeros((size, size), dtype=np.uint8)
        cv2.fillPoly(fill, [np.array(polygon, dtype=np.int32)], 1)
        entry: dict[str, Any] = {
            "index": index,
            "area_px": int(fill.sum()),
            "polygon": [[x, y] for x, y in polygon],
        }
        if layer in ("room", "wall"):
            entry["sides"] = len(polygon)
        entries.append(entry)
    return _write_layer_json(
        path,
        {"size": size, "min_area_px": min_area_px, "components": entries},
    )


def fit_polygons_from_mask(
    layer: StructureLayer,
    mask: np.ndarray,
    *,
    size: int,
    min_area_px: int,
    min_sides: int = 4,
    max_sides: int = 15,
) -> list[PolygonRing]:
    components: list[PolygonRing] = []
    for component in _filtered_components(mask, min_area_px=min_area_px):
        polygon: PolygonRing | None = None
        if layer == "room":
            fit = _fit_room_polygon(
                component,
                width=size,
                height=size,
                min_sides=min_sides,
                max_sides=max_sides,
            )
            if fit is not None:
                polygon = [(point[0], point[1]) for point in fit["polygon"]]
        elif layer == "wall":
            fit = _fit_wall_polygon(
                component,
                width=size,
                height=size,
                min_sides=min_sides,
                max_sides=max_sides,
            )
            if fit is not None:
                polygon = [(point[0], point[1]) for point in fit["polygon"]]
        else:
            rect = _fit_opening_rectangle(component)
            if rect is not None:
                polygon = [(point[0], point[1]) for point in rect["corners"]]
        if polygon is not None and len(polygon) >= 3:
            components.append(polygon)
    return components


def save_final_mask_polygons(
    out_dir: Path,
    masks_by_layer: dict[str, np.ndarray],
    *,
    size: int,
    min_area_px: int,
    min_sides: int = 4,
    max_sides: int = 15,
) -> dict[str, Path]:
    """Write room/wall/door/window polygon JSON files into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    room_components = _filtered_components(masks_by_layer["room"], min_area_px=min_area_px)
    room_entries: list[dict[str, Any]] = []
    for index, component in enumerate(room_components):
        fit = _fit_room_polygon(
            component,
            width=size,
            height=size,
            min_sides=min_sides,
            max_sides=max_sides,
        )
        if fit is None:
            continue
        room_entries.append(
            {
                "index": index,
                "area_px": int(component.sum()),
                **fit,
            }
        )
    paths["room"] = _write_layer_json(
        out_dir / "room.json",
        {"size": size, "min_area_px": min_area_px, "components": room_entries},
    )

    wall_components = _filtered_components(masks_by_layer["wall"], min_area_px=min_area_px)
    wall_entries: list[dict[str, Any]] = []
    for index, component in enumerate(wall_components):
        fit = _fit_wall_polygon(
            component,
            width=size,
            height=size,
            min_sides=min_sides,
            max_sides=max_sides,
        )
        if fit is None:
            continue
        wall_entries.append(
            {
                "index": index,
                "area_px": int(component.sum()),
                **fit,
            }
        )
    paths["wall"] = _write_layer_json(
        out_dir / "wall.json",
        {"size": size, "min_area_px": min_area_px, "components": wall_entries},
    )

    for layer in ("door", "window"):
        components = _filtered_components(masks_by_layer[layer], min_area_px=min_area_px)
        entries: list[dict[str, Any]] = []
        for index, component in enumerate(components):
            rect = _fit_opening_rectangle(component)
            if rect is None:
                continue
            entries.append(
                {
                    "index": index,
                    "area_px": int(component.sum()),
                    **rect,
                }
            )
        paths[layer] = _write_layer_json(
            out_dir / f"{layer}.json",
            {"size": size, "min_area_px": min_area_px, "components": entries},
        )

    return paths
