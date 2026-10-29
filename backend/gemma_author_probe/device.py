"""Resolve GPU (CUDA), then MPS, then CPU once per process."""

from __future__ import annotations

import os

# OpenMP catches SIGINT unless this is set before torch loads it.
os.environ.setdefault("KMP_HANDLE_SIGNALS", "FALSE")

import torch

_device: torch.device | None = None
_announced = False


def _choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device(*, announce: bool = False) -> torch.device:
    global _device, _announced
    if _device is None:
        _device = _choose_device()
    if announce and not _announced:
        print(f"device: {_device}", flush=True)
        _announced = True
    return _device


def release_device_memory() -> None:
    """Return cached GPU/MPS allocations in this process (safe anytime)."""
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        return
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        torch.mps.empty_cache()
