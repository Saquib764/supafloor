"""Write PNG renders and score markdown sidecars for saved plans."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from floorplan.evaluator import score_label
from floorplan.renderer import render_png
from floorplan.schemas.models import FloorPlan, ScoreBreakdown

_RENDER_LAYERS = ["boundary", "rooms", "labels", "doors", "windows", "topology"]


def write_plan_artifacts(
    plan: FloorPlan,
    scores: ScoreBreakdown,
    directory: Path,
    *,
    plan_id: str,
    size: int = 800,
) -> dict[str, str]:
    """Render PNG and write score sidecars. Returns filenames."""
    directory.mkdir(parents=True, exist_ok=True)
    stem = plan_id
    png_name = f"{stem}.png"
    score_md_name = f"{stem}.score.md"
    score_json_name = f"{stem}.score.json"

    (directory / png_name).write_bytes(render_png(plan, size=size, layers=_RENDER_LAYERS))

    per_plan_lines = [
        f"# Score report — {plan.name}",
        "",
        f"Source: `../plans/{stem}.json`",
        "",
        f"![{plan.name}]({png_name})",
        "",
        scores.summary,
        "",
        scores.report,
    ]
    (directory / score_md_name).write_text("\n".join(per_plan_lines))

    hard_failures = [r.rule_id for r in scores.rules if r.hard and not r.passed]
    score_json: dict[str, Any] = {
        "id": plan_id,
        "plan_name": plan.name,
        "overall": scores.overall,
        "summary": scores.summary,
        "categories": {
            "geometry": scores.geometry,
            "topology": scores.topology,
            "construction": scores.construction,
            "privacy": scores.privacy,
            "circulation": scores.circulation,
            "daylight": scores.daylight,
            "architecture": scores.architecture,
            "aesthetics": scores.aesthetics,
        },
        "hard_failures": hard_failures,
        "report_file": score_md_name,
        "image": png_name,
    }
    (directory / score_json_name).write_text(json.dumps(score_json, indent=2))
    return {"png": png_name, "score_md": score_md_name, "score_json": score_json_name}


def score_row(plan: FloorPlan, scores: ScoreBreakdown, *, score_md_name: str, png_name: str | None) -> dict[str, Any]:
    """One index row for README / scores_index.json."""
    hard_failures = [r.rule_id for r in scores.rules if r.hard and not r.passed]
    return {
        "id": plan.metadata.get("plan_id") or plan.id,
        "plan_name": plan.name,
        "overall": scores.overall,
        "summary": scores.summary,
        "rating": score_label(scores.overall),
        "valid": len(hard_failures) == 0,
        "hard_failures": hard_failures,
        "report_file": score_md_name,
        "image": png_name,
    }
