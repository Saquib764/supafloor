#!/usr/bin/env python3
"""Plot hardcoded Florence <loc_*> point pairs onto a 1000×1000 image.

Usage (from backend/florence_trainer):

    uv run python dev/plot_locs.py
"""

from __future__ import annotations

import re
from pathlib import Path

from PIL import Image, ImageDraw

SIZE = 1000
AS_LINE = True
OUT_PATH = Path(__file__).resolve().parent / "output" / "locs.png"

LOC_TEXT = (
    "</s><s><loc_71><loc_81><loc_74><loc_80><loc_217><loc_80><loc_219><loc_81>"
    "<loc_223><loc_80><loc_353><loc_80><loc_356><loc_81><loc_609><loc_80>"
    "<loc_611><loc_81><loc_687><loc_80><loc_691><loc_80><loc_885><loc_80>"
    "<loc_888><loc_81><loc_889><loc_83><loc_891><loc_271><loc_891><loc_274>"
    "<loc_889><loc_275><loc_888><loc_279><loc_888><loc_408><loc_864><loc_408>"
    "<loc_864><loc_401><loc_743><loc_400><loc_743><loc_386><loc_743><loc_383>"
    "<loc_602><loc_382><loc_601><loc_378><loc_601><loc_374><loc_599><loc_370>"
    "<loc_598><loc_366><loc_595><loc_362><loc_595><loc_359><loc_593><loc_355>"
    "<loc_591><loc_352><loc_589><loc_348><loc_586><loc_344><loc_583><loc_340>"
    "<loc_579><loc_335><loc_569><loc_325><loc_563><loc_321><loc_559><loc_318>"
    "<loc_555><loc_317><loc_552><loc_314><loc_548><loc_312><loc_545><loc_310>"
    "<loc_541><loc_309><loc_537><loc_307><loc_533><loc_306><loc_526><loc_305>"
    "<loc_514><loc_305><loc_511><loc_307><loc_508><loc_310><loc_508><loc_422>"
    "<loc_511><loc_425><loc_511><loc_427><loc_472><loc_425><loc_472><loc_416>"
    "<loc_472><loc_412><loc_472><loc_411><loc_473><loc_410><loc_472><loc_332>"
    "<loc_472><loc_329><loc_94><loc_329><loc_94><loc_325><loc_94><loc_332>"
    "<loc_94><loc_328><loc_94><loc_325><loc_93><loc_324><loc_94><loc_321>"
    "<loc_94><loc_165><loc_94><loc_161><loc_94><loc_158><loc_94><loc_157>"
    "<loc_94><loc_104><loc_94><loc_102><loc_94><loc_99><loc_93><loc_98>"
    "<loc_92><loc_95><loc_92><loc_92><loc_93><loc_90><loc_94><loc_86>"
    "<loc_94><loc_83><sep><loc_94><loc_426><loc_387><loc_425><loc_390>"
    "<loc_422><loc_390><loc_414><loc_94><loc_414><sep><loc_94><loc_858>"
    "<loc_432><loc_858><loc_432><loc_884><loc_197><loc_884><loc_197><loc_862>"
    "<loc_197><loc_858><loc_337><loc_858><loc_340><loc_858><loc_337><loc_859>"
    "<loc_337><loc_880><loc_337><loc_884><loc_196><loc_884><loc_196><loc_863>"
    "<loc_197><loc_859><loc_196><loc_858><loc_94><loc_858></s>"
)

_LOC_RE = re.compile(r"<loc_(\d+)>")


def parse_segments(text: str) -> list[list[tuple[int, int]]]:
    segments: list[list[tuple[int, int]]] = []
    for chunk in text.split("<sep>"):
        locs = [int(v) for v in _LOC_RE.findall(chunk)]
        points = [(locs[i], locs[i + 1]) for i in range(0, len(locs) - 1, 2)]
        if points:
            segments.append(points)
    return segments


def main() -> None:
    segments = parse_segments(LOC_TEXT)
    image = Image.new("RGB", (SIZE, SIZE), "white")
    draw = ImageDraw.Draw(image)
    n = 0
    for points in segments:
        n += len(points)
        if AS_LINE and len(points) >= 2:
            draw.line(points, fill="black", width=1)
        else:
            r = 2
            for x, y in points:
                draw.ellipse((x - r, y - r, x + r, y + r), fill="black")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUT_PATH)
    mode = "line" if AS_LINE else "points"
    print(f"wrote {OUT_PATH} ({n} points, {mode})")


if __name__ == "__main__":
    main()
