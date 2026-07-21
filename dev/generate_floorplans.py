#!/usr/bin/env python3
"""Generate floor plans into a target folder as JSON + PNG images.

Usage (from repo root):

    uv run --project backend/floorplan python dev/generate_floorplans.py --out dev/output --count 5
    uv run --project backend/floorplan python dev/generate_floorplans.py --out /tmp/plans --rooms 3 --count 3

Each sample writes:
  <out>/001.json   — floor plan (+ metadata)
  <out>/001.png    — rendered image
  <out>/001.score.md / .score.json — score report sidecars
  <out>/index.json — listing of generated files
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
    parser.add_argument("--rooms", type=int, default=None, help="Sleeping room count (default: cycle 1–4)")
    parser.add_argument("--area", type=float, default=None, help="Target encapsulating area (ft²)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed base")
    parser.add_argument("--no-mutate", action="store_true", help="Skip mutation step")
    parser.add_argument("--size", type=int, default=800, help="PNG render size")
    args = parser.parse_args(argv)

    service_root = Path(__file__).resolve().parents[1] / "backend" / "floorplan"
    if str(service_root) not in sys.path:
        sys.path.insert(0, str(service_root))

    from floorplan.constants import pick_target_area
    from floorplan.dataset import generate_sample
    from floorplan.dataset.artifacts import write_plan_artifacts

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    index: list[dict] = []
    width = max(3, len(str(args.count)))

    for i in range(args.count):
        sample_id = f"{i + 1:0{width}d}"
        rooms = args.rooms if args.rooms is not None else 1 + (i % 4)
        sample = generate_sample(
            rooms,
            target_area=args.area if args.area is not None else pick_target_area(index=i),
            seed=args.seed + i,
            mutate=not args.no_mutate,
            include_renders=False,
        )
        plan = sample.floor_plan
        scores = sample.scores
        assert scores is not None

        files = write_plan_artifacts(plan, scores, out, plan_id=sample_id, size=args.size)

        json_path = out / f"{sample_id}.json"
        payload = {
            "id": sample_id,
            "total_rooms": sample.requirements.get("total_rooms"),
            "floor_plan": plan.model_dump(),
            "boundary": sample.boundary.model_dump() if sample.boundary else None,
            "requirements": sample.requirements,
            "mutation_history": sample.mutation_history,
            "metadata": sample.metadata,
            "files": {"json": json_path.name, **files},
        }
        json_path.write_text(json.dumps(payload, indent=2))

        index.append(
            {
                "id": sample_id,
                "total_rooms": sample.requirements.get("total_rooms"),
                "name": plan.name,
                "json": json_path.name,
                "png": files["png"],
                "score_md": files["score_md"],
                "valid": sample.metadata.get("valid"),
                "room_count": len(plan.rooms),
            }
        )
        print(f"Wrote {json_path.name} + {files['png']} + {files['score_md']} ({plan.name}, rooms={rooms})")

    (out / "index.json").write_text(json.dumps(index, indent=2))
    print(f"\nGenerated {len(index)} plans → {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
