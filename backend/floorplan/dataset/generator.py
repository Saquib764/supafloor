"""Dataset generator pipeline."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Optional

from floorplan.evaluator import hard_failures, score_plan
from floorplan.mutations import mutate_pipeline
from floorplan.renderer import render_plan
from floorplan.schemas.enums import MutationOp
from floorplan.schemas.models import DatasetSample
from floorplan.services.boundary_adapt import adapt_to_boundary, generate_boundary
from floorplan.services.reconcile import reconcile_plan
from floorplan.templates import get_template, instantiate_template, list_templates
from floorplan.topology import (
    constraint_graph_from_plan,
    geometry_hash,
    room_ratio_signature,
    topology_hash,
)

# Target total floor areas (sqft). Generation picks one at random unless overridden.
TARGET_AREAS_SQFT: tuple[int, ...] = (900, 2000, 2500, 3000, 4000, 5000, 6000)

_SQFT_TO_M2 = 0.09290304
# L-boundary footprint vs bounding rect (see generate_boundary shape="l").
_L_FOOTPRINT_FACTOR = 0.7975


def dims_from_area_sqft(
    sqft: float,
    *,
    aspect: float = 1.25,
    boundary_shape: str = "rect",
) -> tuple[float, float]:
    """Return (width_m, depth_m) so the plan footprint ≈ ``sqft`` square feet."""
    area_m2 = float(sqft) * _SQFT_TO_M2
    if boundary_shape == "l":
        area_m2 /= _L_FOOTPRINT_FACTOR
    aspect = max(0.6, min(2.0, float(aspect)))
    width = (area_m2 * aspect) ** 0.5
    depth = area_m2 / width
    return width, depth


def resolve_plan_dims(
    *,
    seed: int,
    width: float | None = None,
    depth: float | None = None,
    target_area_sqft: int | None = None,
    boundary_shape: str = "rect",
) -> tuple[float, float, int]:
    """Resolve width/depth and target area (sqft) for generation.

    When both ``width`` and ``depth`` are set, they win and footprint sqft is derived.
    Otherwise a target area is taken from ``target_area_sqft`` or randomly from
    ``TARGET_AREAS_SQFT``, then missing dimensions are derived.
    """
    rng = random.Random(seed)
    if width is not None and depth is not None:
        factor = _L_FOOTPRINT_FACTOR if boundary_shape == "l" else 1.0
        sqft = int(round(width * depth * factor / _SQFT_TO_M2))
        return float(width), float(depth), sqft

    sqft = int(target_area_sqft) if target_area_sqft is not None else int(rng.choice(TARGET_AREAS_SQFT))

    aspect = 1.15 + rng.random() * 0.30  # ~1.15–1.45
    if width is None and depth is None:
        w, d = dims_from_area_sqft(sqft, aspect=aspect, boundary_shape=boundary_shape)
        return w, d, sqft

    area_m2 = sqft * _SQFT_TO_M2
    if boundary_shape == "l":
        area_m2 /= _L_FOOTPRINT_FACTOR
    if width is None:
        return area_m2 / float(depth), float(depth), sqft
    return float(width), area_m2 / float(width), sqft


def generate_sample(
    template_id: str = "compact_2bed",
    *,
    width: float | None = None,
    depth: float | None = None,
    target_area_sqft: int | None = None,
    boundary_shape: str = "rect",
    seed: int = 0,
    mutate: bool = True,
    include_renders: bool = True,
) -> DatasetSample:
    """Choose Template → Boundary → Fit → Reconcile → Mutate (+ reconcile) → Evaluate → Render.

    Total footprint area defaults to a random choice from ``TARGET_AREAS_SQFT``
    unless ``width``+``depth`` or ``target_area_sqft`` is provided.
    """
    template = get_template(template_id)
    width, depth, area_sqft = resolve_plan_dims(
        seed=seed,
        width=width,
        depth=depth,
        target_area_sqft=target_area_sqft,
        boundary_shape=boundary_shape,
    )
    plan = instantiate_template(template_id, width=width, depth=depth)
    boundary = generate_boundary(width=width, depth=depth, shape=boundary_shape)
    plan = adapt_to_boundary(plan, boundary)
    plan = reconcile_plan(plan)

    history: list[dict[str, Any]] = []
    if mutate:
        plan, history, _ = mutate_pipeline(
            plan,
            ops=template.mutation_rules or list(MutationOp)[:5],
            seed=seed,
            require_valid=True,
        )
        # mutate_pipeline already reconciles accepted mutations

    scores = score_plan(plan)
    topology = constraint_graph_from_plan(plan)
    failures = hard_failures(scores.rules)

    svg = None
    png_b64 = None
    if include_renders:
        rendered = render_plan(plan)
        svg = rendered["svg"]
        png_b64 = rendered["png"]

    return DatasetSample(
        boundary=plan.ensure_boundary(),
        requirements={
            "template_id": template.id,
            "template_name": template.name,
            "rooms": [r.model_dump() for r in template.rooms],
            "target_area_sqft": area_sqft,
        },
        topology=topology,
        floor_plan=plan,
        scores=scores,
        rule_breakdown=scores.rules,
        mutation_history=history,
        svg=svg,
        png_b64=png_b64,
        metadata={
            "seed": seed,
            "target_area_sqft": area_sqft,
            "width_m": round(width, 3),
            "depth_m": round(depth, 3),
            "topology_hash": topology_hash(plan),
            "geometry_hash": geometry_hash(plan),
            "room_ratio_signature": room_ratio_signature(plan),
            "hard_failures": [r.rule_id for r in failures],
            "valid": len(failures) == 0,
        },
    )


def generate_dataset(
    n: int = 10,
    template_ids: Optional[list[str]] = None,
    *,
    seed: int = 0,
    out_dir: Optional[str | Path] = None,
    include_renders: bool = False,
) -> list[DatasetSample]:
    template_ids = template_ids or [t["id"] for t in list_templates()]
    samples: list[DatasetSample] = []
    seen_hashes: set[str] = set()

    attempts = 0
    while len(samples) < n and attempts < n * 5:
        tid = template_ids[attempts % len(template_ids)]
        shape = "rect" if attempts % 3 else "l"
        sample = generate_sample(
            tid,
            boundary_shape=shape if tid != "studio" else "rect",
            seed=seed + attempts,
            mutate=True,
            include_renders=include_renders,
        )
        attempts += 1
        ghash = sample.metadata.get("geometry_hash")
        thash = sample.metadata.get("topology_hash")
        key = f"{thash}:{ghash}"
        if key in seen_hashes:
            continue
        # Prefer valid samples but allow some invalid for diversity of failures
        seen_hashes.add(key)
        samples.append(sample)

    if out_dir:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        for sample in samples:
            (path / f"{sample.id}.json").write_text(sample.model_dump_json(indent=2))
            if sample.svg:
                (path / f"{sample.id}.svg").write_text(sample.svg)
        index = [
            {
                "id": s.id,
                "template": s.requirements.get("template_id"),
                "target_area_sqft": s.metadata.get("target_area_sqft"),
                "overall": s.scores.overall if s.scores else None,
                "valid": s.metadata.get("valid"),
                "topology_hash": s.metadata.get("topology_hash"),
            }
            for s in samples
        ]
        (path / "index.json").write_text(json.dumps(index, indent=2))

    return samples
