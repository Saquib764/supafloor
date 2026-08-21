#!/usr/bin/env python3
"""Parse hardcoded Florence loc text into room polygons and render them.

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

from loc import parse_rooms
from plot import draw_rooms

SIZE = 1000
BINS = 1000
OUT_PATH = Path(__file__).resolve().parent / "output" / "locs.png"

LOC_TEXT = """
</s><s>Kitchen
<loc_726><loc_182>
<poly>
<loc_618><loc_106>
<loc_878><loc_310>
<loc_933><loc_47>
<loc_837><loc_30>
</poly>
<sep>
Bathroom
<loc_767><loc_689>
<poly><loc_611><loc_733>
<loc_768><loc_735>
<loc_768><loc_804>
<loc_931><loc_822>
<loc_862><loc_616>
</poly>

<sep>
Living
<loc_309><loc_499>
<poly> <loc_60><loc_555>
<loc_74><loc_607>
<loc_74><loc_608>
<loc_502><loc_607>
<loc_502><loc_538>
<loc_506><loc_738>
<loc_612><loc_606>
<loc_932><loc_606>
<loc_878><loc_227>
<loc_879><loc_310>
<loc_932><loc_310>
<loc_618><loc_23>
<loc_77><loc_99>
<loc_505><loc_319>
<loc_505><loc_320>
<loc_78><loc_98>
</poly> Bathroom
<loc_383><loc_782>
<poly>

<loc_74><loc_607>

<loc_504><loc_915>
<loc_703><loc_741>
<loc_502><loc_608>
</poly>.
Dining
<loc_798><loc_863>
<poly> Toilet
<loc_696><loc_843>
<loc_726><loc_915>
<loc_726><loc_807>
<loc_768><loc_818>
<loc_768><loc_912>
<loc_848><loc_929>
<loc_946><loc_930>
<loc_931><loc_656>
</poly> <sep>
Room 2
<loc_406><loc_206>
<poly> Bathroom Closet
<loc_707><loc_60>
<poly><loc_0><loc_0>

<loc_505><loc_70>
<loc_629><loc_48>
<loc_629><loc_106>
<loc_837><loc_31>
<loc_933><loc_130>
<loc_878><loc_61>
<loc_950><loc_30>
<loc_505><loc_903>
<loc_505><loc_915>
</poly></s>
"""


def main() -> None:
    rooms = parse_rooms(LOC_TEXT, image_size=SIZE, bins=BINS)
    image = Image.new("RGB", (SIZE, SIZE), "white")
    overlay = draw_rooms(image, rooms)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(OUT_PATH)
    print(f"wrote {OUT_PATH} ({len(rooms)} rooms)")
    for room in rooms:
        print(f"  {room['name']}: {len(room['polygon'])} verts center={room['center']}")


if __name__ == "__main__":
    main()
