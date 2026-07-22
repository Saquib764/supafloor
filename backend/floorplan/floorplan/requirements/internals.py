"""In-memory generation workspace — not persisted in canonical plan JSON."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass
class GenerationInternals:
    """Algorithm workspace used only during generation."""

    parameters: dict[str, Any]
    generated: dict[str, Any]
    plan_id: str = field(default_factory=lambda: uuid4().hex[:12])

    @classmethod
    def from_sample(
        cls,
        total_rooms: int = 3,
        *,
        target_area: float | None = None,
        seed: int = 0,
        plan_id: str | None = None,
    ) -> GenerationInternals:
        from floorplan.requirements.context import run_generation, sample_parameters

        parameters = sample_parameters(total_rooms, target_area=target_area, seed=seed)
        generated = run_generation(parameters)
        return cls(
            parameters=parameters,
            generated=generated,
            plan_id=plan_id or uuid4().hex[:12],
        )
