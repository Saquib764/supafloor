#!/usr/bin/env python3
"""Render infer output JSON onto the source floor-plan image.

Usage (from backend/florence_trainer):

    uv run python dev/render_rooms.py --image test/fp1.jpg
    uv run python dev/render_rooms.py --json dev/room.json --image test/fp1.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import CONFIG_PATH, load_config
from data import prepare_image
from loc import parse_locs
from plot import save_rooms_renderings
from pre_clean import pre_clean

_DEV_DIR = Path(__file__).resolve().parent


def _point_in_polygon(x: float, y: float, ring: list[tuple[float, float]]) -> bool:
    inside = False
    j = len(ring) - 1
    for i, (xi, yi) in enumerate(ring):
        xj, yj = ring[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _attach_sample_points(
    rooms: list[dict],
    prompts: list[str],
    *,
    image_size: int,
    bins: int,
) -> None:
    prompt_points: list[list[int]] = []
    for prompt in prompts:
        locs = parse_locs(prompt, image_size=image_size, bins=bins)
        if locs:
            prompt_points.append(locs[0])

    used: set[int] = set()
    for room in rooms:
        if room.get("sample_point"):
            continue
        polygon = room.get("polygon") or []
        if len(polygon) < 3:
            continue
        ring = [(float(x), float(y)) for x, y in polygon]
        for index, point in enumerate(prompt_points):
            if index in used:
                continue
            if _point_in_polygon(float(point[0]), float(point[1]), ring):
                room["sample_point"] = point
                used.add(index)
                break


def render(json_path: Path, image_path: Path, *, config_path: Path = CONFIG_PATH) -> tuple[Path, Path, list[dict]]:
    cfg = load_config(config_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    rooms = payload.get("rooms") or []
    _attach_sample_points(
        rooms,
        payload.get("prompts") or [],
        image_size=cfg.image_size,
        bins=cfg.loc_bins,
    )
    image = prepare_image(pre_clean(Image.open(image_path)), cfg.image_size)
    overlay, strip = save_rooms_renderings(image, rooms, json_path)
    return overlay, strip, rooms


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw rooms from infer JSON onto a floor-plan image")
    parser.add_argument("--json", type=Path, default=_DEV_DIR / "room.json", help="Infer output JSON")
    parser.add_argument("--image", type=Path, required=True, help="Source floor-plan image used for infer")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="Config for image_size (default config.yaml)")
    args = parser.parse_args()

    overlay, strip, rooms = render(args.json, args.image, config_path=args.config)
    print(f"wrote {overlay.relative_to(_ROOT)} ({len(rooms)} rooms)")
    print(f"wrote {strip.relative_to(_ROOT)}")
    for room in rooms:
        name = room.get("name") or "?"
        polygon = room.get("polygon") or []
        center = room.get("center")
        sample_point = room.get("sample_point")
        print(f"  {name}: {len(polygon)} verts center={center} sample={sample_point}")


if __name__ == "__main__":
    main()
