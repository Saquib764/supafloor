"""Service-layer facades for API and tooling."""

from .boundary_adapt import adapt_to_boundary, generate_boundary

__all__ = ["adapt_to_boundary", "generate_boundary", "FloorPlanEngine"]


def __getattr__(name: str):
    if name == "FloorPlanEngine":
        from .engine import FloorPlanEngine

        return FloorPlanEngine
    raise AttributeError(name)
