"""Dataset generator — generate FloorPlan, persist canonical plan JSON."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Optional

from config import load_config
from schemas.constants import pick_target_area
from dataset.artifacts import write_plan_artifacts
from dataset.store import delete_document, list_documents, load_document, plan_dir, save_document
from evaluator import hard_failures, score_plan
from renderer import render_plan
from requirements import generate_plan_document
from requirements.plan_io import document_to_plan, plan_document_id
from schemas.models import DatasetSample
from topology import (
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
        write_plan_artifacts(plan, scores, plan_dir(plan_id), plan_id=plan_id)

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
    bedroom_count: int,
    *,
    target_area: float | None,
    seed: int,
    include_renders: bool,
    write_artifacts: bool,
) -> tuple[dict[str, Any], DatasetSample]:
    document = generate_plan_document(bedroom_count, target_area=target_area, seed=seed)
    sample = _sample_from_saved(document, include_renders=include_renders, write_artifacts=write_artifacts)
    return document, sample


def _level_score(sample: DatasetSample, level: str) -> float:
    if sample.scores is None:
        return 0.0
    if sample.scores.levels and level in sample.scores.levels:
        return float(sample.scores.levels[level])
    return float(sample.scores.overall)


def _build_sample_meeting_threshold(
    bedroom_count: int,
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
            bedroom_count,
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


def _corpus_attempt(
    attempt_index: int,
    bedroom_count: int | None,
    seed: int,
    attempts_budget: int,
    score_level: str,
    score_threshold: float,
    max_attempts: int,
) -> tuple[dict[str, Any], DatasetSample] | None:
    """One outer corpus slot. Returns None when the level threshold is not met."""
    rooms = bedroom_count if bedroom_count is not None else 1 + (attempt_index % 4)
    document, sample = _build_sample_meeting_threshold(
        rooms,
        target_area=pick_target_area(index=attempt_index),
        seed=seed + attempt_index * attempts_budget,
        include_renders=False,
        write_artifacts=False,
        level=score_level,
        threshold=score_threshold,
        max_attempts=max_attempts,
    )
    if _level_score(sample, score_level) < score_threshold:
        return None
    return document, sample


def generate_sample(
    bedroom_count: int = 3,
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
        bedroom_count,
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
    bedroom_count: int | None = None,
    seed: int = 0,
    out_dir: Optional[str | Path] = None,
    include_renders: bool = False,
    score_level: str | None = None,
    score_threshold: float | None = None,
    max_attempts: int | None = None,
    workers: int = 1,
    compact_json: bool = True,
) -> list[DatasetSample]:
    """Generate up to ``n`` unique plans that meet the score-level threshold.

    Only deterministic pipeline config is used (callers must keep LLM providers off).
    ``workers > 1`` evaluates outer seeds concurrently but accepts results in seed
    order, so the corpus matches a sequential run.
    """
    cfg = load_config()
    level = str(cfg["score_level"]) if score_level is None else score_level
    threshold = float(cfg["score_threshold"]) if score_threshold is None else score_threshold
    attempts_budget = int(cfg["generate_max_attempts"]) if max_attempts is None else int(max_attempts)
    worker_count = max(1, int(workers))

    samples: list[DatasetSample] = []
    seen_hashes: set[str] = set()
    path = Path(out_dir) if out_dir else None
    if path:
        path.mkdir(parents=True, exist_ok=True)

    json_kwargs: dict[str, Any] = {"separators": (",", ":")} if compact_json else {"indent": 2}
    index: list[dict[str, Any]] = []
    max_outer = max(n * 50, n)
    next_attempt = 0
    # Futures keyed by attempt index; consumed in ascending order for determinism.
    pending: dict[int, Any] = {}

    def _submit(executor: ProcessPoolExecutor | None, attempt_index: int) -> None:
        args = (
            attempt_index,
            bedroom_count,
            seed,
            attempts_budget,
            level,
            threshold,
            attempts_budget,
        )
        if executor is None:
            pending[attempt_index] = _corpus_attempt(*args)
        else:
            pending[attempt_index] = executor.submit(_corpus_attempt, *args)

    def _take(attempt_index: int) -> tuple[dict[str, Any], DatasetSample] | None:
        item = pending.pop(attempt_index)
        if worker_count == 1:
            return item
        return item.result()

    executor: ProcessPoolExecutor | None = None
    try:
        if worker_count > 1:
            executor = ProcessPoolExecutor(max_workers=worker_count)

        while len(samples) < n and (next_attempt < max_outer or pending):
            pipeline = 1 if worker_count == 1 else worker_count * 2
            while len(pending) < pipeline and next_attempt < max_outer and len(samples) < n:
                _submit(executor, next_attempt)
                next_attempt += 1

            if not pending:
                break

            attempt_index = min(pending)
            result = _take(attempt_index)
            if result is None:
                continue

            document, sample = result
            key = f"{sample.metadata.get('topology_hash')}:{sample.metadata.get('geometry_hash')}"
            if key in seen_hashes:
                continue
            seen_hashes.add(key)

            if include_renders:
                sample = _sample_from_saved(document, include_renders=True, write_artifacts=False)

            samples.append(sample)
            if path:
                (path / f"{sample.id}.json").write_text(json.dumps(document, **json_kwargs))
                if sample.svg:
                    (path / f"{sample.id}.svg").write_text(sample.svg)
                index.append(
                    {
                        "id": sample.id,
                        "bedroom_count": sample.requirements.get("bedroom_count"),
                        "level": level,
                        "level_score": round(_level_score(sample, level), 4),
                        "overall": sample.scores.overall if sample.scores else None,
                        "valid": sample.metadata.get("valid"),
                        "topology_hash": sample.metadata.get("topology_hash"),
                    }
                )
                if len(samples) % 100 == 0 or len(samples) == n:
                    (path / "index.json").write_text(json.dumps(index, indent=2))
                    print(f"corpus {len(samples)}/{n} (outer_seed_slots={next_attempt})", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    if path:
        (path / "index.json").write_text(json.dumps(index, indent=2))
        meta = {
            "count": len(samples),
            "requested": n,
            "seed": seed,
            "score_level": level,
            "score_threshold": threshold,
            "generate_max_attempts": attempts_budget,
            "workers": worker_count,
            "bedroom_count": bedroom_count,
            "pipeline": "deterministic",
        }
        (path / "manifest.json").write_text(json.dumps(meta, indent=2))

    if len(samples) < n:
        raise RuntimeError(
            f"Only produced {len(samples)}/{n} unique plans meeting "
            f"{level}>={threshold} within {max_outer} outer attempts"
        )

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
