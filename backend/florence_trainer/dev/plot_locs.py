#!/usr/bin/env python3
"""Parse Florence loc text from dev/locs.txt into room polygons and render them.

Usage (from backend/florence_trainer):

    uv run python dev/plot_locs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from loc import parse_locs, parse_rooms
from plot import draw_points, draw_rooms

_DEV_DIR = Path(__file__).resolve().parent
SIZE = 1000
BINS = 1000
MODE = "rooms"  # "rooms" | "point"
LOC_TEXT_PATH = _DEV_DIR / "locs.txt"
OUT_PATH = _DEV_DIR / "output" / "locs.png"


def main() -> None:
    loc_text = LOC_TEXT_PATH.read_text(encoding="utf-8")
    image = Image.new("RGB", (SIZE, SIZE), "white")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if MODE == "point":
        points = parse_locs(loc_text, image_size=SIZE, bins=BINS)
        overlay = draw_points(image, points)
        overlay.save(OUT_PATH)
        print(f"wrote {OUT_PATH} ({len(points)} points)")
        return

    rooms = parse_rooms(loc_text, image_size=SIZE, bins=BINS)
    overlay = draw_rooms(image, rooms)
    overlay.save(OUT_PATH)
    print(f"wrote {OUT_PATH} ({len(rooms)} rooms)")
    for room in rooms:
        print(f"  {room['name']}: {len(room['polygon'])} verts center={room['center']}")


if __name__ == "__main__":
    main()
