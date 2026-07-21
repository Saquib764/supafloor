"""Weighted overall scoring with human-readable descriptive reports."""

from __future__ import annotations

from floorplan.evaluator.furniture import score_furniture
from floorplan.evaluator.rules import evaluate_rules, hard_failures
from floorplan.geometry import area
from floorplan.schemas.models import FloorPlan, RuleResult, ScoreBreakdown

DEFAULT_WEIGHTS = {
    "geometry": 0.15,
    "topology": 0.10,
    "construction": 0.10,
    "privacy": 0.15,
    "circulation": 0.10,
    "furniture": 0.10,
    "daylight": 0.15,
    "architecture": 0.15,
}

RULE_CATEGORY = {
    "polygon_validity": "geometry",
    "room_overlap": "geometry",
    "boundary_containment": "geometry",
    "minimum_area": "geometry",
    "minimum_width": "geometry",
    "rectangularity": "construction",
    "aspect_quality": "architecture",
    "reachability": "topology",
    "required_rooms": "topology",
    "doors_on_walls": "topology",
    "bathroom_accessible": "topology",
    "kitchen_accessible": "topology",
    "windows_on_exterior": "daylight",
    "natural_light": "daylight",
    "privacy": "privacy",
    "circulation": "circulation",
    "living_size": "architecture",
    "bedroom_ratio": "architecture",
    "wet_core": "construction",
    "storage": "architecture",
    "balcony": "architecture",
    "furniture": "furniture",
}

CATEGORY_LABELS = {
    "geometry": "Geometry",
    "topology": "Topology",
    "construction": "Construction",
    "privacy": "Privacy",
    "circulation": "Circulation",
    "furniture": "Furniture usability",
    "daylight": "Daylight",
    "architecture": "Architecture",
}


def score_label(score: float) -> str:
    """Map a 0–10 score to a short human label."""
    if score >= 9.0:
        return "Excellent"
    if score >= 7.5:
        return "Good"
    if score >= 6.0:
        return "Fair"
    if score >= 4.0:
        return "Weak"
    return "Poor"


def _avg(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 5.0


def _plan_blurb(plan: FloorPlan | None) -> str:
    if plan is None:
        return ""
    rooms = ", ".join(f"{r.name} ({area(r.coords()):.0f} sqft)" for r in plan.rooms)
    return (
        f"**Plan:** {plan.name}  \n"
        f"**Rooms ({len(plan.rooms)}):** {rooms}  \n"
        f"**Doors:** {len(plan.doors)} line openings · **Windows:** {len(plan.windows)} line openings\n"
    )


def format_score_report(
    scores: ScoreBreakdown,
    *,
    plan: FloorPlan | None = None,
    title: str | None = None,
) -> str:
    """Build a human-readable Markdown score report."""
    failures = [r for r in scores.rules if r.hard and not r.passed]
    soft_warns = [r for r in scores.rules if not r.hard and r.score < 7.0]
    soft_ok = [r for r in scores.rules if not r.hard and r.score >= 7.0]

    heading = title or (plan.name if plan else "Floor plan score")
    lines: list[str] = [
        f"## {heading}",
        "",
        f"**Overall score: {scores.overall:.2f} / 10 — {score_label(scores.overall)}**",
        "",
    ]
    blurb = _plan_blurb(plan)
    if blurb:
        lines.extend([blurb, ""])

    if failures:
        lines.append("### Hard failures (plan rejected by geometry/topology rules)")
        for r in failures:
            repair = f" — *Repair:* {r.repair_suggestion}" if r.repair_suggestion else ""
            lines.append(f"- **{r.name}** ({r.rule_id}): {r.reason}{repair}")
        lines.append("")
    else:
        lines.append("### Hard rules")
        lines.append("All hard rules passed — the plan is geometrically/topologically acceptable.")
        lines.append("")

    lines.append("### Category scores")
    lines.append("")
    lines.append("| Category | Score | Rating | Weight |")
    lines.append("|---|---:|---|---:|")
    for key in DEFAULT_WEIGHTS:
        val = getattr(scores, key)
        lines.append(
            f"| {CATEGORY_LABELS.get(key, key)} | {val:.2f} | {score_label(val)} | "
            f"{scores.weights.get(key, DEFAULT_WEIGHTS[key]):.2f} |"
        )
    lines.append("")

    lines.append("### Rule details")
    lines.append("")
    for r in scores.rules:
        kind = "HARD" if r.hard else "soft"
        status = "PASS" if r.passed else "FAIL"
        lines.append(
            f"- **{r.name}** [{kind}/{status}] — **{r.score:.1f}/10** ({score_label(r.score)}): {r.reason}"
        )
        if r.repair_suggestion:
            lines.append(f"  - Suggestion: {r.repair_suggestion}")
    lines.append("")

    if soft_warns:
        lines.append("### Areas to improve")
        for r in sorted(soft_warns, key=lambda x: x.score):
            lines.append(f"- {r.name}: {r.reason}" + (f" → {r.repair_suggestion}" if r.repair_suggestion else ""))
        lines.append("")

    if soft_ok and not failures:
        strengths = sorted(soft_ok, key=lambda x: x.score, reverse=True)[:5]
        lines.append("### Strengths")
        for r in strengths:
            lines.append(f"- {r.name} ({r.score:.1f}): {r.reason}")
        lines.append("")

    lines.append(
        f"_Summary: {build_summary(scores, plan=plan)}_"
    )
    return "\n".join(lines)


def build_summary(scores: ScoreBreakdown, *, plan: FloorPlan | None = None) -> str:
    """One-paragraph human summary of the evaluation."""
    failures = [r for r in scores.rules if r.hard and not r.passed]
    weak = [r for r in scores.rules if not r.hard and r.score < 6.0]
    strong = [r for r in scores.rules if not r.hard and r.score >= 8.0]

    plan_bits = []
    if plan is not None:
        plan_bits.append(f"“{plan.name}” with {len(plan.rooms)} rooms")
        total_area = sum(area(r.coords()) for r in plan.rooms)
        plan_bits.append(f"{total_area:.0f} sqft total room area")

    head = "This floor plan scores "
    head += f"**{scores.overall:.2f}/10 ({score_label(scores.overall)})**"
    if plan_bits:
        head += " for " + ", ".join(plan_bits)
    head += "."

    if failures:
        names = ", ".join(r.name for r in failures)
        mid = f" It fails {len(failures)} hard rule(s): {names}."
    else:
        mid = " It passes all hard validity rules."

    if strong:
        mid += " Strong areas include " + ", ".join(r.name.lower() for r in strong[:3]) + "."
    if weak:
        mid += " Weaker areas include " + ", ".join(r.name.lower() for r in weak[:3]) + "."

    cats = {
        "geometry": scores.geometry,
        "privacy": scores.privacy,
        "daylight": scores.daylight,
        "circulation": scores.circulation,
        "furniture": scores.furniture,
    }
    best = max(cats, key=cats.get)
    worst = min(cats, key=cats.get)
    mid += (
        f" Highest category is {CATEGORY_LABELS[best].lower()} ({cats[best]:.1f}); "
        f"lowest is {CATEGORY_LABELS[worst].lower()} ({cats[worst]:.1f})."
    )
    return head + mid


def score_plan(plan: FloorPlan, weights: dict[str, float] | None = None) -> ScoreBreakdown:
    weights = dict(weights or DEFAULT_WEIGHTS)
    rules = evaluate_rules(plan)
    furniture = score_furniture(plan)
    rules = list(rules) + [furniture]

    buckets: dict[str, list[float]] = {k: [] for k in DEFAULT_WEIGHTS}
    for r in rules:
        cat = RULE_CATEGORY.get(r.rule_id, "architecture")
        buckets.setdefault(cat, []).append(r.score)

    cats = {k: _avg(buckets.get(k, [])) for k in DEFAULT_WEIGHTS}
    failures = hard_failures(rules)
    penalty = min(4.0, len(failures) * 1.5)
    overall = sum(cats[k] * weights.get(k, 0.0) for k in cats) - penalty
    overall = max(0.0, min(10.0, overall))

    partial = ScoreBreakdown(
        geometry=round(cats["geometry"], 2),
        topology=round(cats["topology"], 2),
        construction=round(cats["construction"], 2),
        privacy=round(cats["privacy"], 2),
        circulation=round(cats["circulation"], 2),
        furniture=round(cats["furniture"], 2),
        daylight=round(cats["daylight"], 2),
        architecture=round(cats["architecture"], 2),
        overall=round(overall, 2),
        weights=weights,
        rules=rules,
    )
    summary = build_summary(partial, plan=plan)
    report = format_score_report(partial, plan=plan)
    return partial.model_copy(update={"summary": summary, "report": report})
