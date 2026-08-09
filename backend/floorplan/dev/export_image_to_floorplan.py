#!/usr/bin/env python3
"""Export image-to-floorplan training pairs from a floorplan_dataset corpus.

Reads canonical plan JSON from ``floorplan_dataset/`` and writes ``N`` variants per plan:
  <out>/input/<id>_<n>.png   — realistic 1024×1024 render (model input)
  <out>/output/<id>_<n>.json — transformed extraction JSON (model target)

Usage (from backend/floorplan):

    uv run python dev/export_image_to_floorplan.py \\
      --dir ../../floorplan_dataset/basic_5000 \\
      --out ../../image-to-floorplan/basic_5000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

N = 5

_SKIP_NAMES = frozenset({"index.json", "manifest.json", "scores_index.json"})


def _iter_plan_json_paths(directory: Path):
    from dataset.store import PLAN_JSON

    for path in sorted(directory.iterdir()):
        if path.is_dir():
            json_path = path / PLAN_JSON
            if json_path.is_file():
                yield json_path
            continue
        if path.suffix != ".json":
            continue
        if path.name in _SKIP_NAMES or path.name.endswith(".score.json"):
            continue
        yield path


def main(argv: list[str] | None = None) -> int:
    service_root = Path(__file__).resolve().parents[1]
    repo_root = service_root.parents[1]

    parser = argparse.ArgumentParser(
        description="Export image-to-floorplan input PNGs + output extraction JSON"
    )
    parser.add_argument(
        "--dir",
        "-d",
        type=Path,
        default=repo_root / "floorplan_dataset" / "basic_5000",
        help="Corpus folder of canonical plan JSON (default: ../../floorplan_dataset/basic_5000)",
    )
    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        default=repo_root / "image-to-floorplan" / "basic_5000",
        help="Output root with input/ and output/ subfolders",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N plans (default: all)",
    )
    args = parser.parse_args(argv)

    if str(service_root) not in sys.path:
        sys.path.insert(0, str(service_root))

    source = args.dir if args.dir.is_absolute() else service_root / args.dir
    if not source.is_dir():
        print(f"Not a directory: {source}", file=sys.stderr)
        return 1

    out_root = args.out if args.out.is_absolute() else service_root / args.out
    input_dir = out_root / "input"
    output_dir = out_root / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    from dataset.extraction import prepare_extraction, render_extraction_png

    json_paths = list(_iter_plan_json_paths(source))
    if args.limit is not None:
        json_paths = json_paths[: args.limit]
    if not json_paths:
        print(f"No plan JSON files found in {source}", file=sys.stderr)
        return 1

    total = len(json_paths) * N
    written = 0
    for json_path in json_paths:
        for variant in range(N):
            rng = random.Random(f"{json_path}:{variant}")
            extracted = prepare_extraction(json_path, rng=rng)
            stem = f"{extracted['id']}_{variant}"
            (output_dir / f"{stem}.json").write_text(json.dumps(extracted, indent=2))
            render_extraction_png(extracted, input_dir / f"{stem}.png", rng=rng, symbol_probability=0.65)
            written += 1
            if written % 100 == 0 or written == total:
                print(f"exported {written}/{total}", flush=True)

    print(f"Wrote {written} pairs ({len(json_paths)} plans × {N}) → {out_root.resolve()}")
    print(f"  input:  {input_dir.resolve()}")
    print(f"  output: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
