#!/usr/bin/env python3
"""Export a realistic floor plan PNG from JSON.

Usage (from backend/floorplan):

    uv run python dev/export_realistic.py --input out/simple1.rooms.json
    uv run python dev/export_realistic.py --input dev/output/plans/14855ae92998.json
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

_DEFAULT_INPUT = Path("out/simple1.rooms.json")


def main(argv: list[str] | None = None) -> int:
    service_root = Path(__file__).resolve().parents[1]
    if str(service_root) not in sys.path:
        sys.path.insert(0, str(service_root))

    from export.render import export_realistic
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

    if "floor_plan" in payload:
        result = export_realistic(plan=FloorPlan.model_validate(payload["floor_plan"]), size=args.size)
    elif "rooms" in payload and "openings" not in payload and payload["rooms"] and "polygon" in payload["rooms"][0]:
        canvas_size = int(payload.get("meta", {}).get("canvas_size", 1024))
        result = export_realistic(rooms=payload["rooms"], canvas_size=canvas_size, size=args.size)
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
