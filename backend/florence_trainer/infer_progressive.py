"""Progressive random-point Florence-2 inference — loop until point lands in polygon."""

from __future__ import annotations

import random
import sys
import termios
import tty
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
import torch
from PIL import Image

from config import CONFIG_PATH, Config, load_config
from data import (
    format_training_prompt,
    prepare_image,
)
from loc import parse_rooms
from model import load_model_and_processor
from plot import draw_points, draw_rooms
from post_process import insert_t_joints, snap_room_points
from pre_clean import pre_clean
from sample.grid import sample_bright_grid_canvas_points

_ROOT = Path(__file__).resolve().parent
_OUT = _ROOT / "dev" / "output" / "progressive-overlay.png"
_GRID_SIZE = 100
_GRID_TRIES_PER_CELL = 5
_EXPAND_PX = 2
_MAX_WHITE_BLACK_RATIO = 0.7
_MAX_POLYGON_DIFF_RATIO = 0.1
_MAX_STALE_ATTEMPTS = 15
AUTOMATIC = False


def _point_in_polygon(x: float, y: float, ring: list[tuple[float, float]]) -> bool:
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


def _perimeter_pixel_counts(image: Image.Image, polygon: list[list[int]]) -> tuple[int, int]:
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


def _white_black_ratio(above: int, below: int) -> float:
    if below == 0:
        return float("inf") if above else 0.0
    return above / below


def _format_perimeter_counts(above: int, below: int) -> str:
    ratio = _white_black_ratio(above, below)
    ratio_text = "inf" if ratio == float("inf") else f"{ratio:.2f}"
    return f">127: {above}, <127: {below} (white/black: {ratio_text})"


def _expand_polygon(
    polygon: list[list[int]],
    *,
    width: int,
    height: int,
    px: int = _EXPAND_PX,
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
    max_diff_ratio: float = _MAX_POLYGON_DIFF_RATIO,
) -> bool:
    ratios = _polygon_diff_ratios(existing, new, width=width, height=height)
    if ratios is None:
        return False
    old_minus_new, new_minus_old = ratios
    return old_minus_new < max_diff_ratio and new_minus_old < max_diff_ratio


def _is_unlabeled(name: str) -> bool:
    return str(name or "").strip().lower() == "unlabeled"


def _perimeter_ratios(
    image: Image.Image,
    polygon: list[list[int]],
    *,
    width: int,
    height: int,
) -> tuple[float, float]:
    norm_above, norm_below = _perimeter_pixel_counts(image, polygon)
    expanded = _expand_polygon(polygon, width=width, height=height)
    exp_above, exp_below = _perimeter_pixel_counts(image, expanded)
    return (
        _white_black_ratio(norm_above, norm_below),
        _white_black_ratio(exp_above, exp_below),
    )


def _passes_perimeter_ratios(
    image: Image.Image,
    polygon: list[list[int]],
    *,
    width: int,
    height: int,
) -> bool:
    norm_ratio, exp_ratio = _perimeter_ratios(
        image,
        polygon,
        width=width,
        height=height,
    )
    return (
        norm_ratio < _MAX_WHITE_BLACK_RATIO
        and exp_ratio < _MAX_WHITE_BLACK_RATIO
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


def _remove_grid_points_inside_polygon(
    grid_points: list[tuple[float, float]],
    polygon: list[list[int]],
    *,
    point_scale: float,
) -> int:
    inside = _grid_points_inside_polygon(grid_points, polygon, point_scale=point_scale)
    if inside:
        _remove_from_grid_pool(grid_points, inside)
    return len(inside)


def _add_or_update_room(
    rooms: list[dict[str, Any]],
    room: dict[str, Any],
    *,
    image: Image.Image,
    width: int,
    height: int,
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
        if intersection_ratio <= _MAX_POLYGON_DIFF_RATIO:
            continue

        existing_name = str(existing.get("name") or "Room")
        existing_count = int(existing.get("count") or 1)
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
            print("  -> skipped: new polygon area is zero")
            continue
        old_minus_new, new_minus_old = ratios
        if old_minus_new > _MAX_POLYGON_DIFF_RATIO:
            rooms.pop(index)
            rooms.append({**room, "count": 1})
            remnant = _old_minus_new_polygon(
                existing["polygon"],
                room["polygon"],
                width=width,
                height=height,
            )
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
                print(
                    f"  -> split remnant: saved old-new as {remnant_name} "
                    f"(count=1, {len(rooms)} rooms total)"
                )
            elif remnant:
                print("  -> split remnant: failed perimeter ratios, not saved")
            else:
                print("  -> split remnant: could not extract polygon, not saved")
            return None

        if not _polygons_match(existing["polygon"], room["polygon"], width=width, height=height):
            print(
                f"  -> rejected: shape mismatch "
                f"(old-new/new={old_minus_new:.2f}, new-old/new={new_minus_old:.2f})"
            )
            continue
        existing["count"] = existing.get("count", 1) + 1
        if _is_unlabeled(existing["name"]) and not _is_unlabeled(room["name"]):
            existing["name"] = room["name"]
        print(
            f"  -> matched: count={existing['count']}, name={existing['name']}"
        )
        return existing if existing["count"] == 3 else None

    saved = {**room, "count": 1}
    rooms.append(saved)
    print(f"no prospective match — saved {room.get('name') or 'Room'} as new room ({len(rooms)} rooms)")
    return None


def _format_room(room: dict[str, Any]) -> str:
    name = str(room.get("name") or "Room")
    count = int(room.get("count") or 1)
    label = f"{name} (x{count})" if count > 1 else name
    polygon = room.get("polygon") or []
    if not polygon:
        return f"{label}: (empty)"
    points = " -> ".join(f"({int(x)}, {int(y)})" for x, y in polygon)
    return f"{label}: {points}"


def _grid_points_inside_polygon(
    grid_points: list[tuple[float, float]],
    polygon: list[list[int]],
    *,
    point_scale: float,
) -> list[tuple[float, float]]:
    ring = [(float(px), float(py)) for px, py in polygon]
    return [
        point
        for point in grid_points
        if _point_in_polygon(point[0] * point_scale, point[1] * point_scale, ring)
    ]


def _remove_from_grid_pool(
    grid_points: list[tuple[float, float]],
    selected: list[tuple[float, float]],
) -> None:
    selected_set = set(selected)
    grid_points[:] = [point for point in grid_points if point not in selected_set]


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


def _drop_cross_white_canvas_points(
    gray: np.ndarray,
    points: list[tuple[float, float]],
    *,
    point_scale: float,
) -> list[tuple[float, float]]:
    kept: list[tuple[float, float]] = []
    for x, y in points:
        px = int(x * point_scale)
        py = int(y * point_scale)
        white_count = _cross_arm_full_white_count(gray, px, py)
        if white_count >= 2:
            continue
        kept.append((x, y))
    return kept


def _pick_canvas_points(
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
        inside = _grid_points_inside_polygon(
            grid_points,
            room["polygon"],
            point_scale=point_scale,
        )
        if inside:
            count = min(3, len(inside))
            picked = rng.sample(inside, count)
            _remove_from_grid_pool(grid_points, picked)
            return picked, "inside-polygon-batch"

    picked = [rng.choice(grid_points)]
    _remove_from_grid_pool(grid_points, picked)
    return picked, "grid"


def _wait_enter() -> None:
    input("Press Enter to continue...")


def _wait_continue_or_esc() -> bool:
    print("Press Enter to continue, Esc to exit...", end=" ", flush=True)
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                print()
                return False
            if ch in ("\r", "\n"):
                print()
                return True
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _generate_for_prompts(
    model: Any,
    processor: Any,
    cfg: Config,
    image: Image.Image,
    prompts: list[str],
) -> list[str]:
    gen_kwargs = {
        "max_new_tokens": cfg.infer.max_new_tokens,
        "num_beams": cfg.infer.num_beams,
        "do_sample": False,
        "use_cache": True,
    }
    if torch.cuda.is_available():
        inputs = processor(
            text=prompts,
            images=[image] * len(prompts),
            return_tensors="pt",
            padding=True,
        )
        inputs = _move_inputs_to_model(inputs, model)
        with torch.inference_mode():
            generated = model.generate(**inputs, **gen_kwargs)
        return processor.batch_decode(generated, skip_special_tokens=False)

    texts: list[str] = []
    for prompt in prompts:
        inputs = processor(text=[prompt], images=[image], return_tensors="pt")
        inputs = _move_inputs_to_model(inputs, model)
        with torch.inference_mode():
            generated = model.generate(**inputs, **gen_kwargs)
        texts.extend(processor.batch_decode(generated, skip_special_tokens=False))
    return texts


def _move_inputs_to_model(inputs: dict[str, Any], model: Any) -> dict[str, Any]:
    param = next(model.parameters())
    device, dtype = param.device, param.dtype
    return {
        key: (
            value.to(device=device, dtype=dtype)
            if torch.is_floating_point(value)
            else value.to(device)
        )
        if hasattr(value, "to")
        else value
        for key, value in inputs.items()
    }


def run(
    cfg: Config,
    image_path: Path,
    *,
    adapter: Path | None = None,
) -> None:
    adapter_path = adapter or cfg.train.session_dir
    model, processor = load_model_and_processor(cfg, adapter=adapter_path)
    if torch.cuda.is_available():
        model.to("cuda")
    elif torch.backends.mps.is_available():
        model.to("mps")

    was_training = model.training
    model.eval()
    try:
        raw = Image.open(image_path)
        image = prepare_image(raw, cfg.image_size)
        grid_image = prepare_image(pre_clean(raw), cfg.image_size)
        grid_gray = np.array(grid_image.convert("L"))
        rng = random.Random(cfg.infer.seed)
        point_scale = cfg.image_size / cfg.canvas_size
        rooms: list[dict[str, Any]] = []

        grid_points = sample_bright_grid_canvas_points(
            rng,
            grid_image,
            n=_GRID_SIZE,
            canvas_size=cfg.canvas_size,
            point_scale=point_scale,
            tries_per_cell=_GRID_TRIES_PER_CELL,
        )
        before_cross = len(grid_points)
        grid_points = _drop_cross_white_canvas_points(
            grid_gray,
            grid_points,
            point_scale=point_scale,
        )
        if before_cross > len(grid_points):
            print(
                f"cross filter: removed {before_cross - len(grid_points)} grid points "
                f"(2+ full-white arms)"
            )
        print(
            f"grid sample: {len(grid_points)} bright points "
            f"from {_GRID_SIZE}x{_GRID_SIZE} cells"
        )
        _OUT.parent.mkdir(parents=True, exist_ok=True)
        if not AUTOMATIC:
            grid_image_points = [
                [int(x * point_scale), int(y * point_scale)] for x, y in grid_points
            ]
            grid_overlay = draw_points(image, grid_image_points, radius=8)
            grid_overlay.save(_OUT)
            print(f"wrote {_OUT.relative_to(_ROOT)}")
            _wait_enter()

        attempt = 0
        stale_attempts = 0
        while grid_points:
            attempt += 1
            print(f"\n--- attempt {attempt} ---")
            print(f"grid pool: {len(grid_points)} points")
            canvas_points, sample_mode = _pick_canvas_points(
                rng,
                grid_points,
                rooms,
                point_scale=point_scale,
            )
            if not canvas_points:
                break
            if sample_mode != "grid":
                before_cross = len(canvas_points)
                canvas_points = _drop_cross_white_canvas_points(
                    grid_gray,
                    canvas_points,
                    point_scale=point_scale,
                )
                if before_cross > len(canvas_points):
                    print(
                        f"cross filter: dropped {before_cross - len(canvas_points)} "
                        f"picked point(s) (2+ full-white arms)"
                    )
                if not canvas_points:
                    continue
            print(
                f"sample mode: {sample_mode} ({len(canvas_points)} point(s)), "
                f"{len(grid_points)} left in pool"
            )
            prompt = format_training_prompt(
                cfg.task_prompt,
                canvas_points,
                canvas_size=cfg.canvas_size,
                bins=cfg.loc_bins,
            )
            text = _generate_for_prompts(model, processor, cfg, image, [prompt])[0]
            if text.startswith(prompt):
                text = text[len(prompt) :]
            print(prompt)
            print(text)

            parsed_rooms = parse_rooms(
                text,
                image_size=cfg.image_size,
                bins=cfg.loc_bins,
            )
            room_metrics: dict[str, tuple[float, float]] = {}
            if not parsed_rooms:
                print("parsed rooms: (none)")
            else:
                for room_index, room in enumerate(parsed_rooms, start=1):
                    label = f"room {room_index}" if len(parsed_rooms) > 1 else "room"
                    print(f"{label}: name={room['name']}")
                    print(f"polygon: {room['polygon']}")
                    expanded = _expand_polygon(
                        room["polygon"],
                        width=image.width,
                        height=image.height,
                    )
                    norm_above, norm_below = _perimeter_pixel_counts(image, room["polygon"])
                    print(f"perimeter pixels {_format_perimeter_counts(norm_above, norm_below)}")
                    exp_above, exp_below = _perimeter_pixel_counts(image, expanded)
                    print(
                        f"expanded perimeter pixels {_format_perimeter_counts(exp_above, exp_below)}"
                    )
                    room_metrics[str(room["polygon"])] = (
                        _white_black_ratio(norm_above, norm_below),
                        _white_black_ratio(exp_above, exp_below),
                    )

            shared_room = parsed_rooms[0] if len(parsed_rooms) == 1 else None
            overlay_rooms: list[dict[str, Any]] = []
            overlay_seen: set[str] = set()
            overlay_points: list[list[int]] = []
            saved_this_batch: set[str] = set()
            valid_detection = False

            for index, (x, y) in enumerate(canvas_points):
                room = parsed_rooms[index] if index < len(parsed_rooms) else shared_room
                point = [int(x * point_scale), int(y * point_scale)]
                if not AUTOMATIC:
                    overlay_points.append(point)
                if room is None:
                    print(f"point {index + 1} {point}: (no room)")
                    continue

                poly_key = str(room["polygon"])
                ring = [(float(px), float(py)) for px, py in room["polygon"]]
                inside = _point_in_polygon(float(point[0]), float(point[1]), ring)
                print(f"point {index + 1} {point}: inside={inside}")

                norm_ratio, exp_ratio = room_metrics[poly_key]
                accepted = (
                    inside
                    and norm_ratio < _MAX_WHITE_BLACK_RATIO
                    and exp_ratio < _MAX_WHITE_BLACK_RATIO
                )
                if accepted:
                    valid_detection = True
                if accepted and poly_key not in saved_this_batch:
                    saved_this_batch.add(poly_key)
                    confirmed = _add_or_update_room(
                        rooms,
                        room,
                        image=image,
                        width=image.width,
                        height=image.height,
                    )
                    if confirmed is not None:
                        removed = _remove_grid_points_inside_polygon(
                            grid_points,
                            confirmed["polygon"],
                            point_scale=point_scale,
                        )
                        print(
                            f"removed {removed} pool points inside "
                            f"{confirmed['name']} (count=3)"
                        )

                if not AUTOMATIC and poly_key not in overlay_seen:
                    overlay_seen.add(poly_key)
                    overlay_rooms.append({**room, "accepted": accepted})

            if not AUTOMATIC:
                overlay = draw_rooms(image, overlay_rooms)
                overlay = draw_points(overlay, overlay_points, radius=12)
                overlay.save(_OUT)
                print(f"wrote {_OUT.relative_to(_ROOT)}")

            if valid_detection:
                stale_attempts = 0
            else:
                stale_attempts += 1
                if stale_attempts >= _MAX_STALE_ATTEMPTS:
                    print(
                        f"no valid detection in {_MAX_STALE_ATTEMPTS} consecutive "
                        f"attempts — exiting"
                    )
                    break

            if not AUTOMATIC and not _wait_continue_or_esc():
                break

        rooms = snap_room_points(rooms)
        rooms = insert_t_joints(rooms)

        for room in rooms:
            print(_format_room(room))

        if rooms:
            all_rooms_overlay = draw_rooms(image, rooms)
            all_rooms_overlay.save(_OUT)
            print(f"wrote {_OUT.relative_to(_ROOT)} ({len(rooms)} room(s))")
    finally:
        if was_training:
            model.train()


if __name__ == "__main__":
    _IMAGE = _ROOT / "test" / "fp02.jpg"
    AUTOMATIC = True
    run(load_config(CONFIG_PATH), _IMAGE)

