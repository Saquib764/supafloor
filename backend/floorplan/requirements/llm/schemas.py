"""Pydantic response schemas for LLM structured output."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SpaceArea(BaseModel):
    name: str = Field(description="Exact space name from the program")
    target_area_sqft: float = Field(description="Target floor area in square feet", gt=0)


class TargetAreasResponse(BaseModel):
    spaces: list[SpaceArea] = Field(description="One entry per space in the program")


class ConnectionGraphResponse(BaseModel):
    edges: list[list[str]] = Field(
        description="Undirected adjacency pairs using exact space names",
    )


class SpaceCenter(BaseModel):
    name: str = Field(description="Exact space name from the program")
    x: float = Field(description="Center x coordinate in feet")
    y: float = Field(description="Center y coordinate in feet")


class CentersResponse(BaseModel):
    centers: list[SpaceCenter] = Field(description="One center point per space in the program")
