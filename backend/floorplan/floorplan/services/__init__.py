"""Service-layer facades for API and tooling."""

__all__ = ["FloorPlanEngine"]


def __getattr__(name: str):
    if name == "FloorPlanEngine":
        from .engine import FloorPlanEngine

        return FloorPlanEngine
    raise AttributeError(name)
