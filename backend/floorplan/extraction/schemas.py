"""Pydantic schemas for vision LLM floor plan extraction."""

from __future__ import annotations

from pydantic import BaseModel, Field, create_model

from extraction.constants import CANVAS_SIZE


class PixelPoint(BaseModel):
    x: int = Field(ge=1, le=CANVAS_SIZE)
    y: int = Field(ge=1, le=CANVAS_SIZE)


class ExtractedRoom(BaseModel):
    name: str
    polygon: list[PixelPoint] = Field(min_length=3)


class ExtractionResponse(BaseModel):
    rooms: list[ExtractedRoom]


def extraction_response_model(canvas_size: int) -> type[BaseModel]:
    point_model = create_model(
        "PixelPoint",
        x=(int, Field(ge=1, le=canvas_size)),
        y=(int, Field(ge=1, le=canvas_size)),
    )
    room_model = create_model(
        "ExtractedRoom",
        name=(str, ...),
        polygon=(list[point_model], Field(min_length=3)),  # type: ignore[valid-type]
    )
    return create_model("ExtractionResponse", rooms=(list[room_model], ...))  # type: ignore[valid-type]
