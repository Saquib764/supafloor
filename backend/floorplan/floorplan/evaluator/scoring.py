"""Weighted overall scoring with human-readable descriptive reports."""

from __future__ import annotations

from floorplan.evaluator.rules import evaluate_rules, hard_failures
from floorplan.geometry import area
from floorplan.schemas.models import FloorPlan, RuleResult, ScoreBreakdown

DEFAULT_WEIGHTS = {
    "geometry": 0.15,
    "topology": 0.10,
    "construction": 0.10,
    "privacy": 0.11,
    "circulation": 0.10,
    "daylight": 0.13,
    "architecture": 0.13,
    "aesthetics": 0.18,
}

SCORE_LEVELS = ("basic", "intermediate", "advance")

# Each tier adds categories and rules on top of prior tiers.
# basic = room shape + spatial positioning only
SCORE_LEVEL_CATEGORY_ADDS: dict[str, frozenset[str]] = {
    "basic": frozenset({"geometry", "construction", "aesthetics"}),
    "intermediate": frozenset({"topology", "architecture", "privacy", "circulation"}),
    "advance": frozenset({"daylight"}),
}

SCORE_LEVEL_RULE_ADDS: dict[str, frozenset[str]] = {
    "basic": frozenset(),
    "intermediate": frozenset({"reachability", "bathroom_accessible", "kitchen_accessible"}),
    "advance": frozenset({"required_rooms", "windows_on_exterior"}),
}

TIER_GATED_RULES = frozenset().union(*SCORE_LEVEL_RULE_ADDS.values())

RULE_CATEGORY = {
    "polygon_validity": "geometry",
    "room_overlap": "geometry",
    "boundary_containment": "geometry",
    "minimum_area": "geometry",
    "minimum_width": "geometry",
    "rectangularity": "construction",
    "shape_quality": "construction",
    "aspect_quality": "construction",
    "reachability": "topology",
    "required_rooms": "topology",
    "doors_on_walls": "topology",
    "bathroom_accessible": "topology",
    "kitchen_accessible": "topology",
    "windows_on_exterior": "daylight",
    "daylight_quality": "daylight",
    "natural_light": "daylight",
    "privacy": "privacy",
    "circulation_quality": "circulation",
    "circulation": "circulation",
    "living_size": "architecture",
    "bedroom_ratio": "architecture",
    "wet_core": "construction",
    "storage": "architecture",
    "balcony": "architecture",
    "living_centrality": "aesthetics",
    "kitchen_public_link": "topology",
    "entry_sequence": "aesthetics",
    "public_near_exit": "aesthetics",
}

CATEGORY_LABELS = {
    "geometry": "Geometry",
    "topology": "Topology",
    "construction": "Construction",
    "privacy": "Privacy",
    "circulation": "Circulation",
    "daylight": "Daylight",
    "architecture": "Architecture",
    "aesthetics": "Aesthetics",
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


def _cumulative_up_to(level: str, adds: dict[str, frozenset[str]]) -> frozenset[str]:
    result: set[str] = set()
    for tier in SCORE_LEVELS:
        result |= adds[tier]
        if tier == level:
            break
    return frozenset(result)


def active_categories(level: str) -> list[str]:
    active = _cumulative_up_to(level, SCORE_LEVEL_CATEGORY_ADDS)
    return [k for k in DEFAULT_WEIGHTS if k in active]


def active_rules(level: str) -> frozenset[str]:
    active_cats = _cumulative_up_to(level, SCORE_LEVEL_CATEGORY_ADDS)
    tier_rules = _cumulative_up_to(level, SCORE_LEVEL_RULE_ADDS)
    active: set[str] = set()
    for rule_id, cat in RULE_CATEGORY.items():
        if cat not in active_cats:
            continue
        if rule_id in TIER_GATED_RULES:
            if rule_id in tier_rules:
                active.add(rule_id)
        else:
            active.add(rule_id)
    return frozenset(active)


def rule_active_at_level(rule_id: str, level: str) -> bool:
    return rule_id in active_rules(level)


def _category_scores(rules: list[RuleResult], level: str) -> dict[str, float]:
    included = active_rules(level)
    buckets: dict[str, list[float]] = {k: [] for k in DEFAULT_WEIGHTS}
    for r in rules:
        if r.rule_id not in included:
            continue
        cat = RULE_CATEGORY.get(r.rule_id, "architecture")
        buckets.setdefault(cat, []).append(r.score)
    return {k: round(_avg(buckets.get(k, [])), 2) for k in DEFAULT_WEIGHTS}


def _penalty_for_level(rules: list[RuleResult], level: str) -> float:
    included = active_rules(level)
    failures = [r for r in hard_failures(rules) if r.rule_id in included]
    return min(4.0, len(failures) * 1.5)


def overall_for_level(
    cats: dict[str, float],
    weights: dict[str, float],
    level: str,
    *,
    penalty: float = 0.0,
) -> float:
    """Weighted overall using only categories active at the given level."""
    active = active_categories(level)
    total_w = sum(weights.get(k, DEFAULT_WEIGHTS[k]) for k in active)
    if total_w <= 0:
        return max(0.0, min(10.0, 5.0 - penalty))
    overall = sum(
        cats[k] * weights.get(k, DEFAULT_WEIGHTS[k]) / total_w for k in active
    ) - penalty
    return max(0.0, min(10.0, round(overall, 2)))


def _plan_blurb(plan: FloorPlan | None) -> str:
    if plan is None:
        return ""
    rooms = ", ".join(f"{r.name} ({area(r.coords()):.1f} ft²)" for r in plan.rooms)
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
        plan_bits.append(f"{total_area:.1f} ft² total room area")

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
        "aesthetics": scores.aesthetics,
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

    advance_cats = _category_scores(rules, "advance")
    level_categories: dict[str, dict[str, float]] = {}
    levels: dict[str, float] = {}
    for level in SCORE_LEVELS:
        cats = _category_scores(rules, level)
        active = active_categories(level)
        level_categories[level] = {k: cats[k] for k in active}
        levels[level] = overall_for_level(
            cats, weights, level, penalty=_penalty_for_level(rules, level)
        )

    partial = ScoreBreakdown(
        geometry=advance_cats["geometry"],
        topology=advance_cats["topology"],
        construction=advance_cats["construction"],
        privacy=advance_cats["privacy"],
        circulation=advance_cats["circulation"],
        daylight=advance_cats["daylight"],
        architecture=advance_cats["architecture"],
        aesthetics=advance_cats["aesthetics"],
        overall=levels["advance"],
        levels=levels,
        level_categories=level_categories,
        weights=weights,
        rules=rules,
    )
    summary = build_summary(partial, plan=plan)
    report = format_score_report(partial, plan=plan)
    return partial.model_copy(update={"summary": summary, "report": report})
