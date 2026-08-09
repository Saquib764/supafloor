"""Write PNG renders and score sidecars for saved plans."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dataset.layout_prompt import build_layout_prompt
from dataset.store import PLAN_JSON, PLAN_PNG, SCORE_JSON, SCORE_MD
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

    (directory / PLAN_PNG).write_bytes(render_png(plan, size=size, layers=_RENDER_LAYERS))

    layout_prompt = build_layout_prompt(plan)
    per_plan_lines = [
        f"# Score report — {plan.name}",
        "",
        f"Source: `{PLAN_JSON}`",
        "",
        f"![{plan.name}]({PLAN_PNG})",
        "",
        scores.summary,
        "",
    ]
    if layout_prompt:
        per_plan_lines.extend(["### Layout prompt", "", layout_prompt, ""])
    per_plan_lines.append(scores.report)
    (directory / SCORE_MD).write_text("\n".join(per_plan_lines))

    hard_failures = [r.rule_id for r in scores.rules if r.hard and not r.passed]
    score_json: dict[str, Any] = {
        "id": plan_id,
        "plan_name": plan.name,
        "overall": scores.overall,
        "summary": scores.summary,
        "layout_prompt": layout_prompt,
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
        "report_file": SCORE_MD,
        "image": PLAN_PNG,
    }
    (directory / SCORE_JSON).write_text(json.dumps(score_json, indent=2))
    return {
        "png": PLAN_PNG,
        "score_md": SCORE_MD,
        "score_json": SCORE_JSON,
        "layout_prompt": layout_prompt,
    }


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
