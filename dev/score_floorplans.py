#!/usr/bin/env python3
"""Score generated floor plans in a target folder and write README reports.

Usage (from repo root):

    uv run --project backend/floorplan python dev/score_floorplans.py --dir dev/output/plans

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
    from floorplan.requirements.plan_io import llm_to_plan
    from floorplan.schemas.models import DatasetSample, FloorPlan

    raw = json.loads(path.read_text())
    if "floor_plan" in raw:
        try:
            return FloorPlan.model_validate(raw["floor_plan"]), raw
        except Exception:
            sample = DatasetSample.model_validate(raw)
            return sample.floor_plan, raw
    if raw.get("schema_version") and raw.get("rooms"):
        return llm_to_plan(raw), raw
    return FloorPlan.model_validate(raw), raw


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score floor plans in a folder; write README + per-plan reports")
    parser.add_argument("--dir", "-d", type=Path, required=True, help="Folder containing plan JSON files")
    parser.add_argument(
        "--score-dir",
        type=Path,
        default=None,
        help="Folder for PNG + score reports (default: sibling score/ when --dir is plans/)",
    )
    parser.add_argument("--readme", type=str, default="README.md", help="Index README filename")
    args = parser.parse_args(argv)

    service_root = Path(__file__).resolve().parents[1] / "backend" / "floorplan"
    if str(service_root) not in sys.path:
        sys.path.insert(0, str(service_root))

    from floorplan.dataset.artifacts import score_row, write_plan_artifacts
    from floorplan.evaluator import score_label, score_plan
    from floorplan.requirements.plan_io import plan_document_id

    target: Path = args.dir
    if not target.is_dir():
        print(f"Not a directory: {target}", file=sys.stderr)
        return 1

    score_out = args.score_dir
    if score_out is None:
        score_out = target.parent / "score" if target.name == "plans" else target
    score_out.mkdir(parents=True, exist_ok=True)

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
        f"Scored **{len(json_files)}** plan(s) from `{target.resolve()}`.",
        f"Reports written to `{score_out.resolve()}`.",
        "",
        "| Plan | Overall | Rating | Valid | Image |",
        "|---|---:|---|---|---|",
    ]

    details: list[str] = ["", "---", "", "# Detailed reports", ""]

    for path in json_files:
        plan, raw = _load_plan(path)
        scores = score_plan(plan)
        sample_id = plan_document_id(raw) or raw.get("id") or plan.id
        files = write_plan_artifacts(plan, scores, score_out, plan_id=path.stem)
        score_md_name = files["score_md"]
        image_rel = files["png"]

        row = score_row(plan, scores, score_md_name=score_md_name, png_name=image_rel)
        row["id"] = sample_id
        index_rows.append(row)

        valid = row["valid"]
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

        print(f"Scored {path.name} → {scores.overall:.2f} ({rating})")

    readme_parts.extend(details)
    readme_path = score_out / args.readme
    readme_path.write_text("\n".join(readme_parts))
    (score_out / "scores_index.json").write_text(json.dumps(index_rows, indent=2))
    print(f"\nWrote {readme_path} and {len(index_rows)} per-plan score reports")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
