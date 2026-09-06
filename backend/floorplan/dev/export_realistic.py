#!/usr/bin/env python3
"""Export a realistic floor plan PNG via the extraction pipeline.

Usage (from backend/floorplan):

    uv run python dev/export_realistic.py
    uv run python dev/export_realistic.py --input dev/output/plans/9044f6156937/floorplan.json
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_DEFAULT_INPUT = Path("dev/output/plans/9044f6156937/floorplan.json")


def main(argv: list[str] | None = None) -> int:
    service_root = Path(__file__).resolve().parents[1]
    if str(service_root) not in sys.path:
        sys.path.insert(0, str(service_root))

    from extraction.walls import prepare_extraction, render_extraction_png

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
        help="Output JPG path (default: <input_stem>_realistic.jpg)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for scale, offset, and label sizes (default: random each run)",
    )
    parser.add_argument(
        "--symbol-probability",
        type=float,
        default=0.5,
        help="Probability [0-1] of drawing a room symbol per eligible room (default: 0)",
    )
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    input_path = args.input if args.input.is_absolute() else service_root / args.input
    extracted = prepare_extraction(
        input_path,
        rng=rng,
        symbol_probability=args.symbol_probability,
    )

    out_path = args.output
    if out_path is None:
        out_path = input_path.with_name(f"{input_path.stem}_realistic.jpg")
    elif not out_path.is_absolute():
        out_path = service_root / out_path

    render_extraction_png(extracted, out_path, rng=rng)
    print(f"Wrote {out_path} (1024x1024)")
    print(
        {
            "id": extracted["id"],
            "room_count": len(extracted["rooms"]),
            "symbol_count": len(extracted.get("symbols") or []),
            "canvas_size": 1024,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
