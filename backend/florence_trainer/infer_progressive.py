"""Progressive random-point Florence-2 inference — interactive CLI."""

from __future__ import annotations

import random
import sys
import termios
import tty
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from data import format_prompt, prepare_image
from config import CONFIG_PATH, Config, load_config
from inference.generate import generate_for_prompts
from inference.grid import sample_bright_grid_canvas_points
from inference.params import infer_params_from_config
from inference.progressive import (
    GRID_SIZE,
    GRID_TRIES_PER_CELL,
    MAX_STALE_ATTEMPTS,
    add_or_update_room,
    drop_cross_white_canvas_points,
    expand_polygon,
    format_perimeter_counts,
    infer_automatic,
    perimeter_pixel_counts,
    pick_canvas_points,
    point_in_polygon,
    remove_grid_points_inside_polygon,
    white_black_ratio,
)
from loc import parse_rooms
from model import load_model_and_processor
from plot import draw_points, draw_rooms
from post_process import insert_t_joints, snap_room_points
from pre_clean import pre_clean

_ROOT = Path(__file__).resolve().parent
_OUT = _ROOT / "dev" / "output" / "progressive-overlay.png"
AUTOMATIC = False


def _format_room(room: dict[str, Any]) -> str:
    name = str(room.get("name") or "Room")
    count = int(room.get("count") or 1)
    label = f"{name} (x{count})" if count > 1 else name
    polygon = room.get("polygon") or []
    if not polygon:
        return f"{label}: (empty)"
    points = " -> ".join(f"({int(x)}, {int(y)})" for x, y in polygon)
    return f"{label}: {points}"


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

    params = infer_params_from_config(cfg)
    was_training = model.training
    model.eval()
    try:
        raw = Image.open(image_path)
        image = prepare_image(raw, params.image_size)

        if AUTOMATIC:
            rooms = infer_automatic(model, processor, params, raw)
            for room in rooms:
                print(_format_room(room))
            if rooms:
                _OUT.parent.mkdir(parents=True, exist_ok=True)
                all_rooms_overlay = draw_rooms(image, rooms)
                all_rooms_overlay.save(_OUT)
                print(f"wrote {_OUT.relative_to(_ROOT)} ({len(rooms)} room(s))")
            return

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
                f"(2+ full-white arms)"
            )
        print(
            f"grid sample: {len(grid_points)} bright points "
            f"from {GRID_SIZE}x{GRID_SIZE} cells"
        )
        _OUT.parent.mkdir(parents=True, exist_ok=True)
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
            canvas_points, sample_mode = pick_canvas_points(
                rng,
                grid_points,
                rooms,
                point_scale=point_scale,
            )
            if not canvas_points:
                break
            if sample_mode != "grid":
                before_cross = len(canvas_points)
                canvas_points = drop_cross_white_canvas_points(
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
            prompt = format_prompt(
                canvas_points,
                canvas_size=params.canvas_size,
                bins=params.loc_bins,
            )

            text = generate_for_prompts(model, processor, params, image, [prompt])[0]
            if text.startswith(prompt):
                text = text[len(prompt) :]
            print(prompt)
            print(text)

            parsed_rooms = parse_rooms(
                text,
                image_size=params.image_size,
                bins=params.loc_bins,
            )
            room_metrics: dict[str, tuple[float, float]] = {}
            if not parsed_rooms:
                print("parsed rooms: (none)")
            else:
                for room_index, room in enumerate(parsed_rooms, start=1):
                    label = f"room {room_index}" if len(parsed_rooms) > 1 else "room"
                    print(f"{label}: name={room['name']}")
                    print(f"polygon: {room['polygon']}")
                    expanded = expand_polygon(
                        room["polygon"],
                        width=image.width,
                        height=image.height,
                    )
                    norm_above, norm_below = perimeter_pixel_counts(image, room["polygon"])
                    print(f"perimeter pixels {format_perimeter_counts(norm_above, norm_below)}")
                    exp_above, exp_below = perimeter_pixel_counts(image, expanded)
                    print(
                        f"expanded perimeter pixels "
                        f"{format_perimeter_counts(exp_above, exp_below)}"
                    )
                    room_metrics[str(room["polygon"])] = (
                        white_black_ratio(norm_above, norm_below),
                        white_black_ratio(exp_above, exp_below),
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
                overlay_points.append(point)
                if room is None:
                    print(f"point {index + 1} {point}: (no room)")
                    continue

                poly_key = str(room["polygon"])
                ring = [(float(px), float(py)) for px, py in room["polygon"]]
                inside = point_in_polygon(float(point[0]), float(point[1]), ring)
                print(f"point {index + 1} {point}: inside={inside}")

                norm_ratio, exp_ratio = room_metrics[poly_key]
                accepted = inside and norm_ratio < 0.7 and exp_ratio < 0.7
                if accepted:
                    valid_detection = True
                if accepted and poly_key not in saved_this_batch:
                    saved_this_batch.add(poly_key)
                    confirmed = add_or_update_room(
                        rooms,
                        room,
                        image=image,
                        width=image.width,
                        height=image.height,
                        verbose=True,
                    )
                    if confirmed is not None:
                        removed = remove_grid_points_inside_polygon(
                            grid_points,
                            confirmed["polygon"],
                            point_scale=point_scale,
                        )
                        print(
                            f"removed {removed} pool points inside "
                            f"{confirmed['name']} (count=3)"
                        )

                if poly_key not in overlay_seen:
                    overlay_seen.add(poly_key)
                    overlay_rooms.append({**room, "accepted": accepted})

            overlay = draw_rooms(image, overlay_rooms)
            overlay = draw_points(overlay, overlay_points, radius=12)
            overlay.save(_OUT)
            print(f"wrote {_OUT.relative_to(_ROOT)}")

            if valid_detection:
                stale_attempts = 0
            else:
                stale_attempts += 1
                if stale_attempts >= MAX_STALE_ATTEMPTS:
                    print(
                        f"no valid detection in {MAX_STALE_ATTEMPTS} consecutive "
                        f"attempts — exiting"
                    )
                    break

            if not _wait_continue_or_esc():
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
