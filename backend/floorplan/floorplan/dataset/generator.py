"""Dataset generator — generate FloorPlan, persist canonical plan JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from floorplan.config import load_config
from floorplan.constants import pick_target_area
from floorplan.dataset.artifacts import write_plan_artifacts
from floorplan.dataset.store import delete_document, list_documents, load_document, plans_dir, save_document, scores_dir
from floorplan.evaluator import hard_failures, score_plan
from floorplan.renderer import render_plan
from floorplan.requirements import generate_plan_document
from floorplan.requirements.plan_io import document_to_plan, plan_document_id
from floorplan.schemas.models import DatasetSample
from floorplan.topology import (
    constraint_graph_from_plan,
    geometry_hash,
    room_ratio_signature,
    topology_hash,
)


def _sample_from_saved(
    document: dict[str, Any],
    *,
    include_renders: bool = True,
    write_artifacts: bool = False,
) -> DatasetSample:
    plan = document_to_plan(document)
    scores = score_plan(plan)
    topology = constraint_graph_from_plan(plan)
    failures = hard_failures(scores.rules)

    svg = None
    png_b64 = None
    if include_renders:
        rendered = render_plan(plan)
        svg = rendered["svg"]
        png_b64 = rendered["png"]

    plan_id = plan_document_id(document)
    if write_artifacts:
        write_plan_artifacts(plan, scores, scores_dir(), plan_id=plan_id)

    summary = document.get("summary") or {}
    requirements = document.get("requirements") or {}
    return DatasetSample(
        id=plan_id,
        boundary=plan.ensure_boundary(),
        requirements={
            "label": summary.get("name"),
            "bedroom_count": requirements.get("bedroom_count"),
            "target_area_sqft": requirements.get("target_area_sqft"),
        },
        topology=topology,
        floor_plan=plan,
        scores=scores,
        rule_breakdown=scores.rules,
        svg=svg,
        png_b64=png_b64,
        metadata={
            "total_rooms": requirements.get("bedroom_count"),
            "target_area": requirements.get("target_area_sqft"),
            "topology_hash": topology_hash(plan),
            "geometry_hash": geometry_hash(plan),
            "room_ratio_signature": room_ratio_signature(plan),
            "hard_failures": [r.rule_id for r in failures],
            "valid": len(failures) == 0,
            "plan_id": plan_id,
        },
    )


def _build_sample(
    total_rooms: int,
    *,
    target_area: float | None,
    seed: int,
    include_renders: bool,
    write_artifacts: bool,
) -> tuple[dict[str, Any], DatasetSample]:
    document = generate_plan_document(total_rooms, target_area=target_area, seed=seed)
    sample = _sample_from_saved(document, include_renders=include_renders, write_artifacts=write_artifacts)
    return document, sample


def _level_score(sample: DatasetSample, level: str) -> float:
    if sample.scores is None:
        return 0.0
    if sample.scores.levels and level in sample.scores.levels:
        return float(sample.scores.levels[level])
    return float(sample.scores.overall)


def _build_sample_meeting_threshold(
    total_rooms: int,
    *,
    target_area: float | None,
    seed: int,
    include_renders: bool,
    write_artifacts: bool,
    level: str | None = None,
    threshold: float | None = None,
    max_attempts: int | None = None,
) -> tuple[dict[str, Any], DatasetSample]:
    """Generate until the active level score meets threshold, or attempts are exhausted."""
    cfg = load_config()
    level = str(cfg["score_level"]) if level is None else level
    threshold = float(cfg["score_threshold"]) if threshold is None else threshold
    attempts = int(cfg["generate_max_attempts"]) if max_attempts is None else max_attempts

    document: dict[str, Any] | None = None
    sample: DatasetSample | None = None
    for attempt in range(max(1, attempts)):
        document, sample = _build_sample(
            total_rooms,
            target_area=target_area,
            seed=seed + attempt,
            include_renders=False,
            write_artifacts=False,
        )
        if _level_score(sample, level) >= threshold:
            break

    assert document is not None and sample is not None
    if include_renders or write_artifacts:
        sample = _sample_from_saved(
            document,
            include_renders=include_renders,
            write_artifacts=write_artifacts,
        )
    return document, sample


def generate_sample(
    total_rooms: int = 3,
    *,
    target_area: float | None = None,
    seed: int = 0,
    include_renders: bool = True,
    save: bool = False,
    score_level: str | None = None,
    score_threshold: float | None = None,
    max_attempts: int | None = None,
) -> DatasetSample:
    """Generate a plan and optionally persist canonical JSON."""
    document, sample = _build_sample_meeting_threshold(
        total_rooms,
        target_area=target_area,
        seed=seed,
        include_renders=include_renders,
        write_artifacts=save,
        level=score_level,
        threshold=score_threshold,
        max_attempts=max_attempts,
    )
    if save:
        save_document(document)
    return sample


def generate_dataset(
    n: int = 10,
    *,
    total_rooms: int | None = None,
    seed: int = 0,
    out_dir: Optional[str | Path] = None,
    include_renders: bool = False,
    score_level: str | None = None,
    score_threshold: float | None = None,
    max_attempts: int | None = None,
) -> list[DatasetSample]:
    samples: list[DatasetSample] = []
    documents: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    attempts_budget = (
        max_attempts if max_attempts is not None else int(load_config()["generate_max_attempts"])
    )

    attempts = 0
    while len(samples) < n and attempts < n * 5:
        rooms = total_rooms if total_rooms is not None else 1 + (attempts % 4)
        document, sample = _build_sample_meeting_threshold(
            rooms,
            target_area=pick_target_area(index=attempts),
            seed=seed + attempts * attempts_budget,
            include_renders=include_renders,
            write_artifacts=False,
            level=score_level,
            threshold=score_threshold,
            max_attempts=max_attempts,
        )
        attempts += 1
        key = f"{sample.metadata.get('topology_hash')}:{sample.metadata.get('geometry_hash')}"
        if key in seen_hashes:
            continue
        seen_hashes.add(key)
        samples.append(sample)
        documents.append(document)

    if out_dir:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        for document, sample in zip(documents, samples, strict=True):
            (path / f"{sample.id}.json").write_text(json.dumps(document, indent=2))
            if sample.svg:
                (path / f"{sample.id}.svg").write_text(sample.svg)
        index = [
            {
                "id": s.id,
                "total_rooms": s.metadata.get("total_rooms"),
                "overall": s.scores.overall if s.scores else None,
                "valid": s.metadata.get("valid"),
                "topology_hash": s.metadata.get("topology_hash"),
            }
            for s in samples
        ]
        (path / "index.json").write_text(json.dumps(index, indent=2))

    return samples


def list_saved_plans() -> list[dict[str, Any]]:
    return list_documents()


def get_saved_plan(plan_id: str) -> dict[str, Any]:
    return load_document(plan_id)


def delete_saved_plan(plan_id: str) -> None:
    delete_document(plan_id)


def load_sample(plan_id: str, *, include_renders: bool = True) -> DatasetSample:
    """Load saved plan JSON and materialize FloorPlan."""
    return _sample_from_saved(load_document(plan_id), include_renders=include_renders)
