"""Load fpn_five_trainer model (five-class structure masks)."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps

_FPN_FIVE_ROOT = Path(__file__).resolve().parents[2] / "fpn_five_trainer"
_ISOLATED = ("config",)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def prepare_bw(image: Image.Image, size: int) -> Image.Image:
    """Grayscale → resize → invert (white walls on black background)."""
    gray = image.convert("L")
    if gray.size != (size, size):
        gray = gray.resize((size, size), Image.Resampling.LANCZOS)
    return ImageOps.invert(gray)


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class MaskChannels:
    background: int = 0
    rooms: int = 1
    walls: int = 2
    doors: int = 3
    windows: int = 4


@dataclass
class FpnFiveRuntime:
    model: Any
    device: torch.device
    channels: MaskChannels

    @classmethod
    def load(cls, *, checkpoint: Path) -> FpnFiveRuntime:
        saved = {name: sys.modules.get(name) for name in _ISOLATED}
        root_str = str(_FPN_FIVE_ROOT.resolve())
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        try:
            sys.modules["config"] = _load_module(
                "fpn_five_trainer.config",
                _FPN_FIVE_ROOT / "config.py",
            )
            model_mod = _load_module(
                "fpn_five_trainer.net.model",
                _FPN_FIVE_ROOT / "net" / "model.py",
            )
            cfg = sys.modules["config"].load_config()
            device = pick_device()
            if not checkpoint.is_file():
                raise FileNotFoundError(f"FPN Five checkpoint not found: {checkpoint}")
            model = model_mod.build_model(cfg)
            state = torch.load(checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(state["model"])
            model.to(device)
            model.eval()
            ch = cfg.mask_channels
            channels = MaskChannels(
                background=ch.background,
                rooms=ch.rooms,
                walls=ch.walls,
                doors=ch.doors,
                windows=ch.windows,
            )
            return cls(model=model, device=device, channels=channels)
        finally:
            for name, module in saved.items():
                if module is not None:
                    sys.modules[name] = module
                else:
                    sys.modules.pop(name, None)

    @torch.inference_mode()
    def infer(self, prepared: Image.Image) -> np.ndarray:
        batch = image_to_tensor(prepared).unsqueeze(0).to(self.device)
        logits = self.model(batch).mask_logits
        return torch.sigmoid(logits[0]).cpu().numpy()
