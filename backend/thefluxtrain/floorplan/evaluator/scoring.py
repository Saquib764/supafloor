"""Weighted overall scoring across architectural categories."""

from __future__ import annotations

from thefluxtrain.floorplan.evaluator.furniture import score_furniture
from thefluxtrain.floorplan.evaluator.rules import evaluate_rules, hard_failures
from thefluxtrain.floorplan.schemas.models import FloorPlan, ScoreBreakdown

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
}


def score_plan(plan: FloorPlan, weights: dict[str, float] | None = None) -> ScoreBreakdown:
    weights = dict(weights or DEFAULT_WEIGHTS)
    rules = evaluate_rules(plan)
    furniture = score_furniture(plan)
    rules = list(rules) + [furniture]

    buckets: dict[str, list[float]] = {k: [] for k in DEFAULT_WEIGHTS}
    for r in rules:
        cat = RULE_CATEGORY.get(r.rule_id, "architecture")
        if r.rule_id == "furniture":
            cat = "furniture"
        buckets.setdefault(cat, []).append(r.score)

    def avg(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 5.0

    geometry = avg(buckets.get("geometry", []))
    topology = avg(buckets.get("topology", []))
    construction = avg(buckets.get("construction", []))
    privacy = avg(buckets.get("privacy", []))
    circulation = avg(buckets.get("circulation", []))
    furniture_s = avg(buckets.get("furniture", []))
    daylight = avg(buckets.get("daylight", []))
    architecture = avg(buckets.get("architecture", []))

    # Hard failure penalty
    failures = hard_failures(rules)
    penalty = min(4.0, len(failures) * 1.5)

    cats = {
        "geometry": geometry,
        "topology": topology,
        "construction": construction,
        "privacy": privacy,
        "circulation": circulation,
        "furniture": furniture_s,
        "daylight": daylight,
        "architecture": architecture,
    }
    overall = sum(cats[k] * weights.get(k, 0.0) for k in cats) - penalty
    overall = max(0.0, min(10.0, overall))

    return ScoreBreakdown(
        geometry=round(geometry, 2),
        topology=round(topology, 2),
        construction=round(construction, 2),
        privacy=round(privacy, 2),
        circulation=round(circulation, 2),
        furniture=round(furniture_s, 2),
        daylight=round(daylight, 2),
        architecture=round(architecture, 2),
        overall=round(overall, 2),
        weights=weights,
        rules=rules,
    )
