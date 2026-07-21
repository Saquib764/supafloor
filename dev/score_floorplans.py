#!/usr/bin/env python3
"""Score generated floor plans in a target folder and write README reports.

Usage:

    uv run --directory backend python ../dev/score_floorplans.py --dir ../dev/output

For each ``*.json`` plan in the folder, writes:
  <id>.score.md   — human-readable Markdown report (image embedded)
  README.md       — index of all scored plans with embedded images + summaries
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_plan(path: Path):
    from floorplan.schemas.models import DatasetSample, FloorPlan

    raw = json.loads(path.read_text())
    if "floor_plan" in raw:
        # Generator payload or DatasetSample-like
        try:
            return FloorPlan.model_validate(raw["floor_plan"]), raw
        except Exception:
            sample = DatasetSample.model_validate(raw)
            return sample.floor_plan, raw
    return FloorPlan.model_validate(raw), raw


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score floor plans in a folder; write README + per-plan reports")
    parser.add_argument("--dir", "-d", type=Path, required=True, help="Folder containing generated JSON (+ PNG)")
    parser.add_argument("--readme", type=str, default="README.md", help="Index README filename")
    args = parser.parse_args(argv)

    backend = Path(__file__).resolve().parents[1] / "backend"
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))

    from floorplan.evaluator import score_plan

    target: Path = args.dir
    if not target.is_dir():
        print(f"Not a directory: {target}", file=sys.stderr)
        return 1

    json_files = sorted(
        p
        for p in target.glob("*.json")
        if p.name not in ("index.json", "scores_index.json") and not p.name.endswith(".score.json")
    )
    if not json_files:
        print(f"No plan JSON files found in {target}", file=sys.stderr)
        return 1

    index_rows: list[dict] = []
    readme_parts: list[str] = [
        "# Floor plan scores",
        "",
        f"Scored **{len(json_files)}** plan(s) in `{target.resolve()}`.",
        "",
        "| Plan | Overall | Rating | Valid | Image |",
        "|---|---:|---|---|---|",
    ]

    details: list[str] = ["", "---", "", "# Detailed reports", ""]

    for path in json_files:
        plan, raw = _load_plan(path)
        scores = score_plan(plan)
        sample_id = raw.get("id") or plan.id
        png_name = raw.get("files", {}).get("png") or f"{path.stem}.png"
        png_path = target / png_name
        image_rel = png_name if png_path.exists() else None

        score_md_name = f"{path.stem}.score.md"
        report_body = scores.report
        # Embed image at top of per-plan report
        per_plan_lines = [
            f"# Score report — {plan.name}",
            "",
            f"Source: `{path.name}`",
            "",
        ]
        if image_rel:
            per_plan_lines += [f"![{plan.name}]({image_rel})", ""]
        per_plan_lines += [
            scores.summary,
            "",
            report_body,
        ]
        (target / score_md_name).write_text("\n".join(per_plan_lines))

        # Also write machine-readable score sidecar
        score_json = {
            "id": sample_id,
            "plan_name": plan.name,
            "overall": scores.overall,
            "summary": scores.summary,
            "categories": {
                "geometry": scores.geometry,
                "topology": scores.topology,
                "construction": scores.construction,
                "privacy": scores.privacy,
                "circulation": scores.circulation,
                "furniture": scores.furniture,
                "daylight": scores.daylight,
                "architecture": scores.architecture,
                "aesthetics": scores.aesthetics,
            },
            "hard_failures": [r.rule_id for r in scores.rules if r.hard and not r.passed],
            "report_file": score_md_name,
            "image": image_rel,
        }
        (target / f"{path.stem}.score.json").write_text(json.dumps(score_json, indent=2))

        valid = len(score_json["hard_failures"]) == 0
        from floorplan.evaluator import score_label

        rating = score_label(scores.overall)
        img_cell = f"![]({image_rel})" if image_rel else "—"
        readme_parts.append(
            f"| [{plan.name}]({score_md_name}) | {scores.overall:.2f} | {rating} | "
            f"{'yes' if valid else 'no'} | {img_cell} |"
        )

        details.extend(
            [
                f"## {plan.name} (`{path.name}`)",
                "",
            ]
        )
        if image_rel:
            details.append(f"![{plan.name}]({image_rel})")
            details.append("")
        details.append(scores.summary)
        details.append("")
        details.append(f"Full report: [{score_md_name}]({score_md_name})")
        details.append("")
        details.append("---")
        details.append("")

        index_rows.append(score_json)
        print(f"Scored {path.name} → {scores.overall:.2f} ({rating})")

    readme_parts.extend(details)
    readme_path = target / args.readme
    readme_path.write_text("\n".join(readme_parts))
    (target / "scores_index.json").write_text(json.dumps(index_rows, indent=2))
    print(f"\nWrote {readme_path} and {len(index_rows)} per-plan score reports")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
