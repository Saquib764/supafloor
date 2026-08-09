#!/usr/bin/env python3
"""Export Prompt→Output training pairs from a folder of plan JSON files.

Usage (from backend/floorplan):

    uv run python dev/export_training.py \
      --dir ../../floorplan_dataset/basic_5000 \
      --out ../../floorplan_dataset/basic_5000_train.jsonl

Each line is one example:
  { id, input: { prompt, schema }, output: [ { name, actual_area_sqft, polygon, … } ] }
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SKIP_NAMES = frozenset({"index.json", "manifest.json", "scores_index.json"})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export layout training JSONL from plan JSON folder")
    parser.add_argument("--dir", "-d", type=Path, required=True, help="Folder of canonical plan JSON files")
    parser.add_argument("--out", "-o", type=Path, required=True, help="Output JSONL path")
    args = parser.parse_args(argv)

    service_root = Path(__file__).resolve().parents[1]
    if str(service_root) not in sys.path:
        sys.path.insert(0, str(service_root))

    from dataset.layout_prompt import build_training_example
    from floorplan.requirements.plan_io import document_to_plan, plan_document_id

    source: Path = args.dir
    if not source.is_dir():
        print(f"Not a directory: {source}", file=sys.stderr)
        return 1

    json_files = sorted(
        p
        for p in source.glob("*.json")
        if p.name not in _SKIP_NAMES and not p.name.endswith(".score.json")
    )
    if not json_files:
        print(f"No plan JSON files found in {source}", file=sys.stderr)
        return 1

    out: Path = args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    with out.open("w", encoding="utf-8") as fh:
        for path in json_files:
            document = json.loads(path.read_text(encoding="utf-8"))
            plan = document_to_plan(document)
            example = build_training_example(plan)
            if example is None:
                skipped += 1
                continue
            if not example["id"]:
                example["id"] = plan_document_id(document) or path.stem
            fh.write(json.dumps(example, separators=(",", ":"), ensure_ascii=False))
            fh.write("\n")
            written += 1
            if written % 500 == 0 or written == len(json_files):
                print(f"exported {written}/{len(json_files)}", flush=True)

    print(f"Wrote {written} examples → {out.resolve()} (skipped {skipped})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
