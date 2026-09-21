"""Polygon exports for final FPN structure masks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from PIL import Image

from util.canvas import fit_to_canvas_placement
from util.mask_clean import morph_erode_dilate
from util.mask_fit import fit_mask_variants, pick_best_variant
from util.room_components import (
    component_masks_from_binary,
    contour_polygons_from_components,
)

PolygonRing = list[tuple[int, int]]
StructureLayer = Literal["room", "wall", "door", "window"]


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


def compute_wall_mask_from_boundary_and_rooms(
    boundary_polygons: list[PolygonRing],
    room_polygons: list[PolygonRing],
    *,
    size: int,
) -> np.ndarray:
    """Wall = floorplan interior with room interiors cut out."""
    mask = np.zeros((size, size), dtype=bool)
    if boundary_polygons:
        boundary_fill = polygons_to_mask(boundary_polygons, size)
        mask[boundary_fill] = True
    if room_polygons:
        room_fill = polygons_to_mask(room_polygons, size)
        mask[room_fill] = False
    return mask


def compute_wall_polygons_from_boundary_and_rooms(
    boundary_polygons: list[PolygonRing],
    room_polygons: list[PolygonRing],
    *,
    size: int,
    min_area_px: int,
    approx_epsilon: float,
) -> list[PolygonRing]:
    wall_mask = compute_wall_mask_from_boundary_and_rooms(
        boundary_polygons,
        room_polygons,
        size=size,
    )
    components = component_masks_from_binary(wall_mask.astype(np.uint8))
    kept = [component for component in components if int(component.sum()) >= min_area_px]
    return contour_polygons_from_components(kept, approx_epsilon=approx_epsilon)


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
        fill = np.zeros((size, size), dtype=np.uint8)
        cv2.fillPoly(fill, [np.array(polygon, dtype=np.int32)], 1)
        entry: dict[str, Any] = {
            "index": index,
            "area_px": int(fill.sum()),
            "polygon": [[x, y] for x, y in polygon],
            "sides": len(polygon),
        }
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
    opening_approx_epsilon: float = 2.0,
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
        elif layer in ("door", "window"):
            polys = contour_polygons_from_components(
                [component],
                approx_epsilon=opening_approx_epsilon,
            )
            if polys:
                polygon = polys[0]
        if polygon is not None and len(polygon) >= 3:
            components.append(polygon)
    return components


def save_final_mask_contour_polygons(
    out_dir: Path,
    *,
    size: int,
    min_area_px: int,
    room_polygons: list[PolygonRing],
    door_polygons: list[PolygonRing],
    window_polygons: list[PolygonRing],
    boundary_polygon: PolygonRing,
    wall_approx_epsilon: float,
) -> dict[str, Path]:
    """Write contour room/door/window polygons and boundary-derived wall into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    boundary = boundary_polygon if len(boundary_polygon) >= 3 else []
    wall_polygons = compute_wall_polygons_from_boundary_and_rooms(
        [boundary] if boundary else [],
        room_polygons,
        size=size,
        min_area_px=min_area_px,
        approx_epsilon=wall_approx_epsilon,
    )

    background_plane = polygons_to_mask([boundary], size) if boundary else np.zeros((size, size), dtype=bool)
    Image.fromarray(background_plane.astype(np.uint8) * 255, mode="L").save(out_dir / "background.png")
    Image.fromarray(polygons_to_mask(room_polygons, size).astype(np.uint8) * 255, mode="L").save(
        out_dir / "room.png"
    )
    Image.fromarray(
        compute_wall_mask_from_boundary_and_rooms(
            [boundary] if boundary else [],
            room_polygons,
            size=size,
        ).astype(np.uint8)
        * 255,
        mode="L",
    ).save(out_dir / "wall.png")
    Image.fromarray(polygons_to_mask(door_polygons, size).astype(np.uint8) * 255, mode="L").save(
        out_dir / "door.png"
    )
    Image.fromarray(polygons_to_mask(window_polygons, size).astype(np.uint8) * 255, mode="L").save(
        out_dir / "window.png"
    )

    paths = {
        "room": save_layer_polygons(
            out_dir / "room.json",
            "room",
            room_polygons,
            size=size,
            min_area_px=min_area_px,
        ),
        "wall": save_layer_polygons(
            out_dir / "wall.json",
            "wall",
            wall_polygons,
            size=size,
            min_area_px=min_area_px,
        ),
        "door": save_layer_polygons(
            out_dir / "door.json",
            "door",
            door_polygons,
            size=size,
            min_area_px=min_area_px,
        ),
        "window": save_layer_polygons(
            out_dir / "window.json",
            "window",
            window_polygons,
            size=size,
            min_area_px=min_area_px,
        ),
        "background": save_layer_polygons(
            out_dir / "background.json",
            "room",
            [boundary] if boundary else [],
            size=size,
            min_area_px=min_area_px,
        ),
    }
    return paths
