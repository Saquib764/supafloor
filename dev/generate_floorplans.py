#!/usr/bin/env python3
"""Generate floor plans into a target folder as canonical plan JSON + PNG images.

Usage (from repo root):

    uv run --project backend/floorplan python dev/generate_floorplans.py --out dev/output/plans --count 5
    uv run --project backend/floorplan python dev/generate_floorplans.py --out /tmp/plans --rooms 3 --count 3

Each sample writes:
  <out>/001.json   — canonical plan JSON
  <score>/001.png  — rendered image (when --score-dir set)
  <score>/001.score.md / .score.json — score report sidecars
  <out>/index.json — listing of generated files
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate floor plans (JSON + PNG) into a folder")
    parser.add_argument("--out", "-o", type=Path, required=True, help="Target output folder for plan JSON")
    parser.add_argument("--score-dir", type=Path, default=None, help="Folder for PNG/score sidecars (default: sibling score/)")
    parser.add_argument("--count", "-n", type=int, default=5, help="Number of samples to generate")
    parser.add_argument("--rooms", type=int, default=None, help="Sleeping room count (default: cycle 1–4)")
    parser.add_argument("--area", type=float, default=None, help="Target encapsulating area (ft²)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed base (runtime only, not saved)")
    parser.add_argument("--size", type=int, default=800, help="PNG render size")
    args = parser.parse_args(argv)

    service_root = Path(__file__).resolve().parents[1] / "backend" / "floorplan"
    if str(service_root) not in sys.path:
        sys.path.insert(0, str(service_root))

    from floorplan.constants import pick_target_area
    from floorplan.dataset.artifacts import write_plan_artifacts
    from floorplan.evaluator import score_plan
    from floorplan.requirements import build_document, document_to_plan
    from floorplan.requirements.plan_io import internal_to_plan_document, plan_document_id

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    score_out = args.score_dir or (out.parent / "score" if out.name == "plans" else out)
    score_out.mkdir(parents=True, exist_ok=True)

    index: list[dict] = []
    width = max(3, len(str(args.count)))

    for i in range(args.count):
        sample_id = f"{i + 1:0{width}d}"
        rooms = args.rooms if args.rooms is not None else 1 + (i % 4)
        internal = build_document(
            rooms,
            target_area=args.area if args.area is not None else pick_target_area(index=i),
            seed=args.seed + i,
            plan_id=sample_id,
        )
        plan = document_to_plan(internal)
        document = internal_to_plan_document(internal, plan)
        scores = score_plan(plan)
        files = write_plan_artifacts(plan, scores, score_out, plan_id=sample_id, size=args.size)

        json_path = out / f"{sample_id}.json"
        json_path.write_text(json.dumps(document, indent=2))

        index.append(
            {
                "id": plan_document_id(document),
                "total_rooms": document.get("requirements", {}).get("bedroom_count"),
                "name": document.get("summary", {}).get("name"),
                "json": json_path.name,
                "png": files["png"],
                "score_md": files["score_md"],
                "valid": len([r for r in scores.rules if r.hard and not r.passed]) == 0,
                "room_count": len(plan.rooms),
            }
        )
        print(f"Wrote {json_path.name} + {files['png']} + {files['score_md']} ({plan.name}, rooms={rooms})")

    (out / "index.json").write_text(json.dumps(index, indent=2))
    print(f"\nGenerated {len(index)} plans → {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
