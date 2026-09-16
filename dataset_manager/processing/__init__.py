"""Floor-plan mask sample loaders for trainer datasets."""

from .adapters import get_adapter
from .types import MaskChannels

__all__ = ["MaskChannels", "get_adapter"]
