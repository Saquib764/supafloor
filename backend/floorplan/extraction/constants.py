"""Shared constants for LLM floor plan image extraction."""

from __future__ import annotations

from typing import Literal

CANVAS_SIZE: int = 1024
EXTRACTION_MODEL: str = "gpt-5.6-sol"
LLM_PROVIDER: Literal["openai", "replicate"] = "openai"
REPLICATE_MODEL: str = (
    "lucataco/gemma-4-31b-it:14c757b42dc096c7d2926c72b40503f5d4acbf1b9b27e910e6722b2a96e51cde"
)
REPLICATE_MAX_NEW_TOKENS: int = 2048
REPLICATE_TEMPERATURE: float = 0.7
REPLICATE_TOP_P: float = 0.95
LLM_TIMEOUT_SECONDS: float = 120.0
MORPH_KERNEL_SIZE: int = 3
MORPH_ITERATIONS: int = 1
THRESHOLD_VALUE: int = 200
