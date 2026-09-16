"""Shared types for floor-plan mask dataset processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from PIL import Image

import numpy as np

_MASK_CHANNEL_FIELDS = ("background", "rooms", "walls", "doors", "windows")
_DEFAULT_MASK_CHANNELS = {
    "background": 0,
    "rooms": 1,
    "walls": 2,
    "doors": 3,
    "windows": 4,
}


@dataclass(frozen=True)
class MaskChannels:
    background: int
    rooms: int
    walls: int
    doors: int
    windows: int

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> MaskChannels:
        values = dict(_DEFAULT_MASK_CHANNELS)
        if raw:
            for name in _MASK_CHANNEL_FIELDS:
                if name in raw:
                    values[name] = int(raw[name])
        return cls(**values)

    @property
    def count(self) -> int:
        return len(_MASK_CHANNEL_FIELDS)

    def indices(self) -> tuple[int, ...]:
        return tuple(getattr(self, name) for name in _MASK_CHANNEL_FIELDS)

    def labels(self) -> tuple[str, ...]:
        ordered = sorted((getattr(self, name), name) for name in _MASK_CHANNEL_FIELDS)
        return tuple(name for _, name in ordered)

    def validate(self, out_channels: int) -> None:
        indices = self.indices()
        if len(set(indices)) != len(indices):
            raise ValueError(f"mask_channels indices must be unique, got {indices}")
        expected = list(range(out_channels))
        if sorted(indices) != expected:
            raise ValueError(
                f"mask_channels must be exactly {expected} for out_channels={out_channels}, "
                f"got {indices}"
            )


class SampleAdapter(Protocol):
    @property
    def hub_columns(self) -> tuple[str, ...]: ...

    @property
    def image_keys(self) -> tuple[str, ...]: ...

    def load_sample(
        self,
        row: dict[str, Any],
        *,
        size: int,
        channels: MaskChannels,
        image_key: str,
    ) -> tuple[Image.Image, np.ndarray]: ...
