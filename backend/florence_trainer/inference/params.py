"""Inference parameters — no trainer Config required."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from config import Config


@dataclass(frozen=True)
class InferParams:
    image_size: int
    canvas_size: int
    loc_bins: int
    max_new_tokens: int
    num_beams: int
    seed: int | None = None

    @property
    def point_scale(self) -> float:
        return self.image_size / self.canvas_size


def infer_params_from_config(cfg: Config) -> InferParams:
    return InferParams(
        image_size=cfg.image_size,
        canvas_size=cfg.canvas_size,
        loc_bins=cfg.loc_bins,
        max_new_tokens=cfg.infer.max_new_tokens,
        num_beams=cfg.infer.num_beams,
        seed=cfg.infer.seed,
    )


def infer_params_from_mapping(raw: dict[str, Any]) -> InferParams:
    return InferParams(
        image_size=int(raw["image_size"]),
        canvas_size=int(raw["canvas_size"]),
        loc_bins=int(raw["loc_bins"]),
        max_new_tokens=int(raw["max_new_tokens"]),
        num_beams=int(raw["num_beams"]),
        seed=int(raw["infer_seed"]) if raw["infer_seed"] is not None else None,
    )
