#!/usr/bin/env python3
"""Extract room polygons from a floor plan image via OpenAI vision LLM.

Usage (from backend/floorplan):

    uv run python dev/extract_floorplan.py
    uv run python dev/extract_floorplan.py --image out/fp.webp --out out
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_DEFAULT_IMAGE = Path("out/fp1.jpeg")
_DEFAULT_OUT = Path("out")


def main(argv: list[str] | None = None) -> int:
    service_root = Path(__file__).resolve().parents[1]
    if str(service_root) not in sys.path:
        sys.path.insert(0, str(service_root))

    parser = argparse.ArgumentParser(description="Extract room polygons from a floor plan image")
    parser.add_argument(
        "--image",
        "-i",
        type=Path,
        default=service_root / _DEFAULT_IMAGE,
        help=f"Input floor plan image (default: {_DEFAULT_IMAGE})",
    )
    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        default=service_root / _DEFAULT_OUT,
        help=f"Output directory (default: {_DEFAULT_OUT})",
    )
    parser.add_argument("--prompt", default="v1", help="Prompt template name (extraction/prompts/<name>.yaml)")
    args = parser.parse_args(argv)

    from extraction import extract_rooms

    if not args.image.is_file():
        print(f"Image not found: {args.image}", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem

    rooms, meta, _plot_png, _debug_png = extract_rooms(
        args.image,
        prompt=args.prompt,
        out_dir=args.out,
        stem=stem,
    )

    json_path = args.out / f"{stem}.rooms.json"
    print(f"Wrote {json_path} ({len(rooms)} rooms)")
    print(f"Wrote {args.out / f'{stem}_out.png'}")
    print(f"Wrote {args.out / f'{stem}_debug.png'}")
    print(f"Model: {meta['model']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
