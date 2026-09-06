#!/usr/bin/env python3
"""Render realistic vs simple PNGs for side-by-side comparison.

Usage (from backend/floorplan):

    uv run python dev/render_compare.py
    uv run python dev/render_compare.py --dir ../../floorplan_dataset/basic_5000 --count 5
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_SKIP_NAMES = frozenset({"index.json", "manifest.json", "scores_index.json"})


def _bootstrap_service_root() -> Path:
    service_root = Path(__file__).resolve().parents[1]
    root = str(service_root)
    while root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)
    return service_root


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
    service_root = _bootstrap_service_root()
    repo_root = service_root.parents[1]

    parser = argparse.ArgumentParser(description="Render realistic and simple PNGs for comparison")
    parser.add_argument(
        "--dir",
        "-d",
        type=Path,
        default=repo_root / "floorplan_dataset" / "basic_5000",
        help="Corpus folder of canonical plan JSON",
    )
    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        default=service_root / "dev" / "output" / "render_compare",
        help="Output directory for comparison JPGs",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of floor plans to render (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed (default: 0)",
    )
    args = parser.parse_args(argv)

    source = args.dir if args.dir.is_absolute() else service_root / args.dir
    if not source.is_dir():
        print(f"Not a directory: {source}", file=sys.stderr)
        return 1

    out_dir = args.out if args.out.is_absolute() else service_root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    from extraction.walls import (
        EXPORT_IMAGE_EXT,
        prepare_extraction,
        render_extraction_png,
        render_extraction_simple_png,
    )

    json_paths = list(_iter_plan_json_paths(source))[: max(1, args.count)]
    if not json_paths:
        print(f"No plan JSON files found in {source}", file=sys.stderr)
        return 1

    for index, json_path in enumerate(json_paths):
        rng = random.Random(args.seed + index)
        extracted = prepare_extraction(json_path, rng=rng, symbol_probability=0.65)
        stem = str(extracted["id"])
        render_extraction_png(extracted, out_dir / f"{stem}_realistic{EXPORT_IMAGE_EXT}", rng=rng)
        render_extraction_simple_png(extracted, out_dir / f"{stem}_simple{EXPORT_IMAGE_EXT}", rng=rng)
        print(f"{index + 1}/{len(json_paths)} {stem}", flush=True)

    print(f"Wrote {len(json_paths) * 2} images → {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
