"""Map registered Hub dataset names to sample loaders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image

import numpy as np

from . import floorplan_mask, manual_mask
from .types import MaskChannels, SampleAdapter

_JSON_COLUMNS = ("stem", "image", "simple_image", "extraction")
_MANUAL_MASK_COLUMNS = (
    "stem",
    "image",
    "background",
    "room",
    "wall",
    "door",
    "window",
)


@dataclass(frozen=True)
class FloorplanMaskAdapter:
    @property
    def hub_columns(self) -> tuple[str, ...]:
        return _JSON_COLUMNS

    @property
    def image_keys(self) -> tuple[str, ...]:
        return ("image", "simple_image")

    def load_sample(
        self,
        row: dict[str, Any],
        *,
        size: int,
        channels: MaskChannels,
        image_key: str,
    ) -> tuple[Image.Image, np.ndarray]:
        return floorplan_mask.load_sample(
            row,
            size=size,
            channels=channels,
            image_key=image_key,
        )


@dataclass(frozen=True)
class ManualMaskAdapter:
    @property
    def hub_columns(self) -> tuple[str, ...]:
        return _MANUAL_MASK_COLUMNS

    @property
    def image_keys(self) -> tuple[str, ...]:
        return ("image",)

    def load_sample(
        self,
        row: dict[str, Any],
        *,
        size: int,
        channels: MaskChannels,
        image_key: str,
    ) -> tuple[Image.Image, np.ndarray]:
        return manual_mask.load_sample(
            row,
            size=size,
            channels=channels,
            image_key=image_key,
        )


_ADAPTERS: dict[str, SampleAdapter] = {
    "floorplan-to-json": FloorplanMaskAdapter(),
    "floorplan-manual-mask": ManualMaskAdapter(),
}


def get_adapter(dataset_name: str) -> SampleAdapter:
    adapter = _ADAPTERS.get(dataset_name)
    if adapter is None:
        known = ", ".join(sorted(_ADAPTERS)) or "(none)"
        raise KeyError(f"Unknown dataset {dataset_name!r}. Supported: {known}")
    return adapter
