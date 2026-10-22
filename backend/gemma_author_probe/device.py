"""Pick CUDA, MPS, or CPU."""

from __future__ import annotations

import torch


def pick_device(*, announce: bool = True) -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    if announce:
        print(f"device: {device}", flush=True)
    return device
