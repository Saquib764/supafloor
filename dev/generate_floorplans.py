#!/usr/bin/env python3
"""Generate floor plans into a target folder as JSON + PNG images.

Usage (from repo root or backend):

    uv run --directory backend python ../dev/generate_floorplans.py --out ../dev/output --n 5
    uv run --directory backend python ../dev/generate_floorplans.py --out /tmp/plans --template compact_2bed --n 3

Each sample writes:
  <out>/<id>.json   — floor plan (+ metadata)
  <out>/<id>.png    — rendered image
  <out>/index.json  — listing of generated files
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate floor plans (JSON + PNG) into a folder")
    parser.add_argument("--out", "-o", type=Path, required=True, help="Target output folder")
    parser.add_argument("--count", "-n", type=int, default=5, help="Number of samples to generate")
    parser.add_argument("--template", default=None, help="Template id (default: cycle all)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed base")
    parser.add_argument("--no-mutate", action="store_true", help="Skip mutation step")
    parser.add_argument("--size", type=int, default=800, help="PNG render size")
    args = parser.parse_args(argv)

    # Allow running without editable install if backend is on PYTHONPATH
    backend = Path(__file__).resolve().parents[1] / "backend"
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))

    from floorplan.dataset import generate_sample
    from floorplan.renderer import render_png
    from floorplan.templates import list_templates

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    template_ids = [args.template] if args.template else [t["id"] for t in list_templates()]
    index: list[dict] = []

    for i in range(args.count):
        tid = template_ids[i % len(template_ids)]
        sample = generate_sample(
            tid,
            seed=args.seed + i,
            mutate=not args.no_mutate,
            include_renders=False,
            width=8.0 + (i % 5),
            depth=7.0 + (i % 4),
            boundary_shape="rect" if i % 3 else "l",
        )
        plan = sample.floor_plan
        png_bytes = render_png(
            plan,
            size=args.size,
            layers=["boundary", "rooms", "labels", "doors", "windows"],
        )

        json_path = out / f"{sample.id}.json"
        png_path = out / f"{sample.id}.png"

        payload = {
            "id": sample.id,
            "template_id": tid,
            "floor_plan": plan.model_dump(),
            "boundary": sample.boundary.model_dump() if sample.boundary else None,
            "requirements": sample.requirements,
            "mutation_history": sample.mutation_history,
            "metadata": sample.metadata,
            "files": {"json": json_path.name, "png": png_path.name},
        }
        json_path.write_text(json.dumps(payload, indent=2))
        png_path.write_bytes(png_bytes)

        index.append(
            {
                "id": sample.id,
                "template_id": tid,
                "name": plan.name,
                "json": json_path.name,
                "png": png_path.name,
                "valid": sample.metadata.get("valid"),
                "room_count": len(plan.rooms),
            }
        )
        print(f"Wrote {json_path.name} + {png_path.name} ({plan.name}, template={tid})")

    (out / "index.json").write_text(json.dumps(index, indent=2))
    print(f"\nGenerated {len(index)} plans → {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
