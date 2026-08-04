"""Pydantic schemas for vision LLM floor plan extraction."""

from __future__ import annotations

from pydantic import BaseModel, Field

from extraction.constants import CANVAS_SIZE


class PixelPoint(BaseModel):
    x: int = Field(ge=1, le=CANVAS_SIZE)
    y: int = Field(ge=1, le=CANVAS_SIZE)


class ExtractedRoom(BaseModel):
    name: str
    polygon: list[PixelPoint] = Field(min_length=3)


class ExtractionResponse(BaseModel):
    rooms: list[ExtractedRoom]
