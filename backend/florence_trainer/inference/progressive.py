"""Progressive and point-guided Florence room inference."""

from __future__ import annotations

import random
from typing import Any, Iterator

import cv2
import numpy as np
from PIL import Image

from data import format_training_prompt, prepare_image
from inference.generate import generate_for_prompts
from inference.grid import sample_bright_grid_canvas_points
from inference.params import InferParams
from loc import parse_rooms
from post_process import insert_t_joints, snap_room_points
from pre_clean import pre_clean

GRID_SIZE = 100
GRID_TRIES_PER_CELL = 5
EXPAND_PX = 2
MAX_WHITE_BLACK_RATIO = 0.7
MAX_POLYGON_DIFF_RATIO = 0.1
MAX_STALE_ATTEMPTS = 15
MAX_POINTS_PER_PROMPT = 6


def point_in_polygon(x: float, y: float, ring: list[tuple[float, float]]) -> bool:
    inside = False
    j = len(ring) - 1
    for i, (xi, yi) in enumerate(ring):
        xj, yj = ring[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _iter_line(x0: int, y0: int, x1: int, y1: int) -> Iterator[tuple[int, int]]:
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        yield x, y
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


def perimeter_pixel_counts(image: Image.Image, polygon: list[list[int]]) -> tuple[int, int]:
    gray = image.convert("L")
    width, height = gray.size
    pixels = gray.load()
    above = 0
    below = 0
    seen: set[tuple[int, int]] = set()
    n = len(polygon)
    for i in range(n):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % n]
        for x, y in _iter_line(int(x0), int(y0), int(x1), int(y1)):
            if (x, y) in seen or not (0 <= x < width and 0 <= y < height):
                continue
            seen.add((x, y))
            value = pixels[x, y]
            if value > 127:
                above += 1
            elif value < 127:
                below += 1
    return above, below


def white_black_ratio(above: int, below: int) -> float:
    if below == 0:
        return float("inf") if above else 0.0
    return above / below


def format_perimeter_counts(above: int, below: int) -> str:
    ratio = white_black_ratio(above, below)
    ratio_text = "inf" if ratio == float("inf") else f"{ratio:.2f}"
    return f">127: {above}, <127: {below} (white/black: {ratio_text})"


def expand_polygon(
    polygon: list[list[int]],
    *,
    width: int,
    height: int,
    px: int = EXPAND_PX,
) -> list[list[int]]:
    if len(polygon) < 3:
        return polygon
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    k = 2 * px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dilated = cv2.dilate(mask, kernel)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return polygon
    contour = max(contours, key=cv2.contourArea)
    return [[int(x), int(y)] for x, y in contour[:, 0]]


def _polygon_mask(polygon: list[list[int]], width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(polygon) >= 3:
        pts = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def _polygon_diff_ratios(
    existing: list[list[int]],
    new: list[list[int]],
    *,
    width: int,
    height: int,
) -> tuple[float, float] | None:
    new_mask = _polygon_mask(new, width, height)
    new_area = int(new_mask.sum())
    if new_area == 0:
        return None
    existing_mask = _polygon_mask(existing, width, height)
    old_minus_new = int((existing_mask & ~new_mask).sum()) / new_area
    new_minus_old = int((new_mask & ~existing_mask).sum()) / new_area
    return old_minus_new, new_minus_old


def _polygons_match(
    existing: list[list[int]],
    new: list[list[int]],
    *,
    width: int,
    height: int,
    max_diff_ratio: float = MAX_POLYGON_DIFF_RATIO,
) -> bool:
    ratios = _polygon_diff_ratios(existing, new, width=width, height=height)
    if ratios is None:
        return False
    old_minus_new, new_minus_old = ratios
    return old_minus_new < max_diff_ratio and new_minus_old < max_diff_ratio


def _is_unlabeled(name: str) -> bool:
    return str(name or "").strip().lower() == "unlabeled"


def _passes_perimeter_ratios(
    image: Image.Image,
    polygon: list[list[int]],
    *,
    width: int,
    height: int,
) -> bool:
    norm_above, norm_below = perimeter_pixel_counts(image, polygon)
    expanded = expand_polygon(polygon, width=width, height=height)
    exp_above, exp_below = perimeter_pixel_counts(image, expanded)
    return (
        white_black_ratio(norm_above, norm_below) < MAX_WHITE_BLACK_RATIO
        and white_black_ratio(exp_above, exp_below) < MAX_WHITE_BLACK_RATIO
    )


def _old_minus_new_polygon(
    existing: list[list[int]],
    new: list[list[int]],
    *,
    width: int,
    height: int,
) -> list[list[int]] | None:
    existing_mask = _polygon_mask(existing, width, height)
    new_mask = _polygon_mask(new, width, height)
    diff = (existing_mask & ~new_mask).astype(np.uint8)
    contours, _ = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return None
    return [[int(x), int(y)] for x, y in contour[:, 0]]


def _intersection_over_old_area(
    existing: list[list[int]],
    new: list[list[int]],
    *,
    width: int,
    height: int,
) -> float:
    existing_mask = _polygon_mask(existing, width, height)
    old_area = int(existing_mask.sum())
    if old_area == 0:
        return 0.0
    new_mask = _polygon_mask(new, width, height)
    return int((existing_mask & new_mask).sum()) / old_area


def grid_points_inside_polygon(
    grid_points: list[tuple[float, float]],
    polygon: list[list[int]],
    *,
    point_scale: float,
) -> list[tuple[float, float]]:
    ring = [(float(px), float(py)) for px, py in polygon]
    return [
        point
        for point in grid_points
        if point_in_polygon(point[0] * point_scale, point[1] * point_scale, ring)
    ]


def remove_from_grid_pool(
    grid_points: list[tuple[float, float]],
    selected: list[tuple[float, float]],
) -> None:
    selected_set = set(selected)
    grid_points[:] = [point for point in grid_points if point not in selected_set]


def remove_grid_points_inside_polygon(
    grid_points: list[tuple[float, float]],
    polygon: list[list[int]],
    *,
    point_scale: float,
) -> int:
    inside = grid_points_inside_polygon(grid_points, polygon, point_scale=point_scale)
    if inside:
        remove_from_grid_pool(grid_points, inside)
    return len(inside)


def add_or_update_room(
    rooms: list[dict[str, Any]],
    room: dict[str, Any],
    *,
    image: Image.Image,
    width: int,
    height: int,
    verbose: bool = False,
) -> dict[str, Any] | None:
    for index, existing in enumerate(rooms):
        if int(existing.get("count") or 1) == 3:
            continue
        intersection_ratio = _intersection_over_old_area(
            existing["polygon"],
            room["polygon"],
            width=width,
            height=height,
        )
        if intersection_ratio <= MAX_POLYGON_DIFF_RATIO:
            continue

        existing_name = str(existing.get("name") or "Room")
        existing_count = int(existing.get("count") or 1)
        if verbose:
            print(
                f"prospective match: {existing_name} (count={existing_count}) "
                f"intersection/old={intersection_ratio:.2f}"
            )

        ratios = _polygon_diff_ratios(
            existing["polygon"],
            room["polygon"],
            width=width,
            height=height,
        )
        if ratios is None:
            if verbose:
                print("  -> skipped: new polygon area is zero")
            continue
        old_minus_new, new_minus_old = ratios
        if old_minus_new > MAX_POLYGON_DIFF_RATIO:
            rooms.pop(index)
            rooms.append({**room, "count": 1})
            remnant = _old_minus_new_polygon(
                existing["polygon"],
                room["polygon"],
                width=width,
                height=height,
            )
            if verbose:
                print(
                    f"  -> split: old-new/new={old_minus_new:.2f} > 0.10 "
                    f"— deleted {existing_name}, saved new as count=1"
                )
            if remnant and _passes_perimeter_ratios(
                image,
                remnant,
                width=width,
                height=height,
            ):
                remnant_name = existing.get("name") or room.get("name") or "Room"
                rooms.append(
                    {
                        "name": remnant_name,
                        "polygon": remnant,
                        "count": 1,
                    }
                )
                if verbose:
                    print(
                        f"  -> split remnant: saved old-new as {remnant_name} "
                        f"(count=1, {len(rooms)} rooms total)"
                    )
            elif verbose:
                if remnant:
                    print("  -> split remnant: failed perimeter ratios, not saved")
                else:
                    print("  -> split remnant: could not extract polygon, not saved")
            return None

        if not _polygons_match(existing["polygon"], room["polygon"], width=width, height=height):
            if verbose:
                print(
                    f"  -> rejected: shape mismatch "
                    f"(old-new/new={old_minus_new:.2f}, new-old/new={new_minus_old:.2f})"
                )
            continue
        existing["count"] = existing.get("count", 1) + 1
        if _is_unlabeled(existing["name"]) and not _is_unlabeled(room["name"]):
            existing["name"] = room["name"]
        if verbose:
            print(f"  -> matched: count={existing['count']}, name={existing['name']}")
        return existing if existing["count"] == 3 else None

    saved = {**room, "count": 1}
    rooms.append(saved)
    if verbose:
        print(
            f"no prospective match — saved {room.get('name') or 'Room'} as new room "
            f"({len(rooms)} rooms)"
        )
    return None


def _arm_is_full_white(arm: np.ndarray) -> bool:
    return arm.size > 0 and float(arm.mean()) >= 250.0


def _cross_arm_full_white_count(gray: np.ndarray, px: int, py: int) -> int:
    height, width = gray.shape
    px = min(max(px, 0), width - 1)
    py = min(max(py, 0), height - 1)
    arms = (
        gray[py, :px],
        gray[py, px:],
        gray[:py, px],
        gray[py:, px],
    )
    return sum(1 for arm in arms if _arm_is_full_white(arm))


def drop_cross_white_canvas_points(
    gray: np.ndarray,
    points: list[tuple[float, float]],
    *,
    point_scale: float,
) -> list[tuple[float, float]]:
    kept: list[tuple[float, float]] = []
    for x, y in points:
        px = int(x * point_scale)
        py = int(y * point_scale)
        if _cross_arm_full_white_count(gray, px, py) >= 2:
            continue
        kept.append((x, y))
    return kept


def pick_canvas_points(
    rng: random.Random,
    grid_points: list[tuple[float, float]],
    rooms: list[dict[str, Any]],
    *,
    point_scale: float,
) -> tuple[list[tuple[float, float]], str]:
    if not grid_points:
        return [], "grid"

    under_confirmed = [room for room in rooms if int(room.get("count") or 1) < 3]
    if under_confirmed:
        room = rng.choice(under_confirmed)
        inside = grid_points_inside_polygon(
            grid_points,
            room["polygon"],
            point_scale=point_scale,
        )
        if inside:
            count = min(3, len(inside))
            picked = rng.sample(inside, count)
            remove_from_grid_pool(grid_points, picked)
            return picked, "inside-polygon-batch"

    picked = [rng.choice(grid_points)]
    remove_from_grid_pool(grid_points, picked)
    return picked, "grid"


def detect_rooms_at_points(
    model: Any,
    processor: Any,
    params: InferParams,
    image: Image.Image,
    canvas_points: list[tuple[float, float]],
    rooms: list[dict[str, Any]],
    *,
    point_scale: float | None = None,
    verbose: bool = False,
) -> tuple[bool, list[dict[str, Any]]]:
    scale = point_scale if point_scale is not None else params.point_scale
    prompt = format_training_prompt(
        params.task_prompt,
        canvas_points,
        canvas_size=params.canvas_size,
        bins=params.loc_bins,
    )
    text = generate_for_prompts(model, processor, params, image, [prompt])[0]
    if text.startswith(prompt):
        text = text[len(prompt) :]

    if verbose:
        print(prompt, flush=True)
        print(text, flush=True)

    parsed_rooms = parse_rooms(
        text,
        image_size=params.image_size,
        bins=params.loc_bins,
    )
    room_metrics: dict[str, tuple[float, float]] = {}
    if verbose and not parsed_rooms:
        print("parsed rooms: (none)", flush=True)
    for room_index, room in enumerate(parsed_rooms, start=1):
        polygon = room.get("polygon") or []
        if len(polygon) < 3:
            continue
        norm_above, norm_below = perimeter_pixel_counts(image, polygon)
        expanded = expand_polygon(
            polygon,
            width=image.width,
            height=image.height,
        )
        exp_above, exp_below = perimeter_pixel_counts(image, expanded)
        room_metrics[str(polygon)] = (
            white_black_ratio(norm_above, norm_below),
            white_black_ratio(exp_above, exp_below),
        )
        if verbose:
            label = f"room {room_index}" if len(parsed_rooms) > 1 else "room"
            print(f"{label}: name={room['name']}", flush=True)
            print(f"polygon: {polygon}", flush=True)
            print(
                f"perimeter pixels {format_perimeter_counts(norm_above, norm_below)}",
                flush=True,
            )
            print(
                f"expanded perimeter pixels "
                f"{format_perimeter_counts(exp_above, exp_below)}",
                flush=True,
            )

    shared_room = parsed_rooms[0] if len(parsed_rooms) == 1 else None
    saved_this_batch: set[str] = set()
    valid_detection = False
    confirmed: list[dict[str, Any]] = []

    for index, (x, y) in enumerate(canvas_points):
        room = parsed_rooms[index] if index < len(parsed_rooms) else shared_room
        point = [int(x * scale), int(y * scale)]
        if room is None:
            if verbose:
                print(f"point {index + 1} {point}: (no room)", flush=True)
            continue

        poly_key = str(room["polygon"])
        ring = [(float(px), float(py)) for px, py in room["polygon"]]
        inside = point_in_polygon(float(point[0]), float(point[1]), ring)
        if verbose:
            print(f"point {index + 1} {point}: inside={inside}", flush=True)
        norm_ratio, exp_ratio = room_metrics.get(poly_key, (float("inf"), float("inf")))
        accepted = (
            inside
            and norm_ratio < MAX_WHITE_BLACK_RATIO
            and exp_ratio < MAX_WHITE_BLACK_RATIO
        )
        if accepted:
            valid_detection = True
        if accepted and poly_key not in saved_this_batch:
            saved_this_batch.add(poly_key)
            matched = add_or_update_room(
                rooms,
                room,
                image=image,
                width=image.width,
                height=image.height,
                verbose=verbose,
            )
            if matched is not None:
                confirmed.append(matched)

    return valid_detection, confirmed


def infer_automatic(
    model: Any,
    processor: Any,
    params: InferParams,
    raw: Image.Image,
) -> list[dict[str, Any]]:
    """Run full progressive grid inference on a floor-plan image."""
    image = prepare_image(raw, params.image_size)
    grid_image = prepare_image(pre_clean(raw), params.image_size)
    grid_gray = np.array(grid_image.convert("L"))
    rng = random.Random(params.seed)
    point_scale = params.point_scale
    rooms: list[dict[str, Any]] = []

    grid_points = sample_bright_grid_canvas_points(
        rng,
        grid_image,
        n=GRID_SIZE,
        canvas_size=params.canvas_size,
        point_scale=point_scale,
        tries_per_cell=GRID_TRIES_PER_CELL,
    )
    before_cross = len(grid_points)
    grid_points = drop_cross_white_canvas_points(
        grid_gray,
        grid_points,
        point_scale=point_scale,
    )
    if before_cross > len(grid_points):
        print(
            f"cross filter: removed {before_cross - len(grid_points)} grid points "
            f"(2+ full-white arms)",
            flush=True,
        )
    print(
        f"grid sample: {len(grid_points)} bright points "
        f"from {GRID_SIZE}x{GRID_SIZE} cells",
        flush=True,
    )

    attempt = 0
    stale_attempts = 0
    while grid_points:
        attempt += 1
        print(f"\n--- attempt {attempt} ---", flush=True)
        print(f"grid pool: {len(grid_points)} points", flush=True)
        canvas_points, sample_mode = pick_canvas_points(
            rng,
            grid_points,
            rooms,
            point_scale=point_scale,
        )
        if not canvas_points:
            break
        if sample_mode != "grid":
            before_picked = len(canvas_points)
            canvas_points = drop_cross_white_canvas_points(
                grid_gray,
                canvas_points,
                point_scale=point_scale,
            )
            if before_picked > len(canvas_points):
                print(
                    f"cross filter: dropped {before_picked - len(canvas_points)} "
                    f"picked point(s) (2+ full-white arms)",
                    flush=True,
                )
            if not canvas_points:
                continue
        print(
            f"sample mode: {sample_mode} ({len(canvas_points)} point(s)), "
            f"{len(grid_points)} left in pool",
            flush=True,
        )

        valid_detection, batch_confirmed = detect_rooms_at_points(
            model,
            processor,
            params,
            image,
            canvas_points,
            rooms,
            point_scale=point_scale,
            verbose=True,
        )

        for confirmed_room in batch_confirmed:
            removed = remove_grid_points_inside_polygon(
                grid_points,
                confirmed_room["polygon"],
                point_scale=point_scale,
            )
            if removed:
                print(
                    f"removed {removed} pool points inside "
                    f"{confirmed_room['name']} (count=3)",
                    flush=True,
                )

        if valid_detection:
            stale_attempts = 0
        else:
            stale_attempts += 1
            if stale_attempts >= MAX_STALE_ATTEMPTS:
                print(
                    f"no valid detection in {MAX_STALE_ATTEMPTS} consecutive "
                    f"attempts — exiting",
                    flush=True,
                )
                break

    rooms = snap_room_points(rooms)
    rooms = insert_t_joints(rooms)
    print(f"done: {len(rooms)} room(s)", flush=True)
    return rooms


def infer_at_points(
    model: Any,
    processor: Any,
    params: InferParams,
    image: Image.Image,
    points: list[tuple[float, float]],
    rooms: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Infer rooms at explicit canvas points and merge into existing rooms."""
    if not points:
        return rooms

    working = [dict(room) for room in rooms]
    batches = [
        points[index : index + MAX_POINTS_PER_PROMPT]
        for index in range(0, len(points), MAX_POINTS_PER_PROMPT)
    ]
    for batch in batches:
        detect_rooms_at_points(
            model,
            processor,
            params,
            image,
            batch,
            working,
        )

    return snap_room_points(working)
