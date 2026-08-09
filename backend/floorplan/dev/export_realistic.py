#!/usr/bin/env python3
"""Export a realistic floor plan PNG via the dataset extraction pipeline.

Usage (from backend/floorplan):

    uv run python dev/export_realistic.py
    uv run python dev/export_realistic.py --input dev/output/plans/9044f6156937/floorplan.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_DEFAULT_INPUT = Path("dev/output/plans/9044f6156937/floorplan.json")


def main(argv: list[str] | None = None) -> int:
    service_root = Path(__file__).resolve().parents[1]
    if str(service_root) not in sys.path:
        sys.path.insert(0, str(service_root))

    from dataset.extraction import prepare_extraction, render_extraction_png

    parser = argparse.ArgumentParser(description="Export realistic floor plan PNG")
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=service_root / _DEFAULT_INPUT,
        help=f"Input floorplan or extraction JSON (default: {_DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output PNG path (default: <input_stem>_realistic.png)",
    )
    parser.add_argument("--size", type=int, default=950, help="Canvas size (default: 950)")
    args = parser.parse_args(argv)

    input_path = args.input if args.input.is_absolute() else service_root / args.input
    extracted = prepare_extraction(input_path)

    out_path = args.output
    if out_path is None:
        out_path = input_path.with_name(f"{input_path.stem}_realistic.png")
    elif not out_path.is_absolute():
        out_path = service_root / out_path

    render_extraction_png(extracted, out_path, canvas_size=args.size)
    print(f"Wrote {out_path} ({args.size}x{args.size})")
    print({"id": extracted["id"], "room_count": len(extracted["rooms"]), "canvas_size": args.size})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
