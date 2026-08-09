#!/usr/bin/env python3
"""Export a realistic floor plan PNG from JSON.

Usage (from backend/floorplan):

    uv run python dev/export_realistic.py
    uv run python dev/export_realistic.py --input out/simple1.rooms.json
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

_DEFAULT_INPUT = Path("dev/output/plans/a5a3eda8662b.json")


def main(argv: list[str] | None = None) -> int:
    service_root = Path(__file__).resolve().parents[1]
    if str(service_root) not in sys.path:
        sys.path.insert(0, str(service_root))

    from export.render import _opening_lines, export_realistic
    from floorplan.schemas.models import FloorPlan

    parser = argparse.ArgumentParser(description="Export realistic floor plan PNG")
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=service_root / _DEFAULT_INPUT,
        help=f"Input JSON (rooms list or floor_plan object, default: {_DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output PNG path (default: <input_stem>_realistic.png)",
    )
    parser.add_argument("--size", type=int, default=1024, help="Output PNG size (default: 1024)")
    args = parser.parse_args(argv)

    input_path = args.input if args.input.is_absolute() else service_root / args.input
    payload = json.loads(input_path.read_text())

    if "rooms" in payload and payload["rooms"] and "polygon" in payload["rooms"][0]:
        openings = payload.get("openings") or {}
        result = export_realistic(
            rooms=payload["rooms"],
            doors=_opening_lines(openings.get("doors") or []),
            windows=_opening_lines(openings.get("windows") or []),
            size=args.size,
        )
    elif "floor_plan" in payload:
        result = export_realistic(plan=FloorPlan.model_validate(payload["floor_plan"]), size=args.size)
    else:
        result = export_realistic(plan=FloorPlan.model_validate(payload), size=args.size)

    out_path = args.output
    if out_path is None:
        out_path = input_path.with_name(f"{input_path.stem}_realistic.png")
    elif not out_path.is_absolute():
        out_path = service_root / out_path

    out_path.write_bytes(base64.b64decode(result["png_b64"]))
    print(f"Wrote {out_path} ({result['size']}x{result['size']})")
    print(result["metadata"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
