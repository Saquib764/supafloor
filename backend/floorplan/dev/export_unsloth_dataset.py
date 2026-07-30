#!/usr/bin/env python3
"""Export created floor plans to an Unsloth-ready Gemma 4 LoRA JSONL dataset.

After generating plans (``dev/generate_floorplans.py`` or ``floorplan dataset``),
run this to wrap prompt / schema / output into Unsloth Studio upload format.

Gemma 4 IT format (LoRA SFT):
  - JSONL with a ``messages`` array
  - Roles: ``system``, ``user``, ``model`` (not ``assistant``)
  - Also writes Alpaca columns ``instruction`` / ``input`` / ``output``
    (mapped from prompt / schema / output) for Studio auto-detect

For the generic Prompt→Output JSONL (no chat wrapper), use ``export_training.py``.

Usage (from backend/floorplan):

    uv run python dev/export_unsloth_dataset.py \\
      --dir dev/output/plans \\
      --out dev/output/unsloth/gemma4_e4b_sft.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export plan JSON folder → Unsloth Gemma-4 LoRA JSONL (prompt/schema/output)"
    )
    parser.add_argument(
        "--dir",
        "-d",
        type=Path,
        required=True,
        help="Folder of canonical plan JSON files (after generation)",
    )
    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        required=True,
        help="Output JSONL path for Unsloth Studio upload",
    )
    parser.add_argument(
        "--format",
        choices=["messages", "alpaca", "both"],
        default="both",
        help="messages=Gemma-4 chat; alpaca=instruction/input/output; both (default)",
    )
    args = parser.parse_args(argv)

    if not args.dir.is_dir():
        print(f"Plans directory not found: {args.dir}", file=sys.stderr)
        return 1

    service_root = Path(__file__).resolve().parents[1]
    if str(service_root) not in sys.path:
        sys.path.insert(0, str(service_root))

    from floorplan.dataset.unsloth import export_unsloth_dataset

    summary = export_unsloth_dataset(args.dir, args.out, fmt=args.format)
    print(json.dumps(summary, indent=2))
    print(
        f"\nUpload {summary['output']} in Unsloth Studio → train "
        f"unsloth/gemma-4-E4B-it (or google/gemma-4-E4B-it) with LoRA."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
