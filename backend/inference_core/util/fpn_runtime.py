"""Load FPN structure and room models from trainer packages (isolated import)."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps

_FPN_ROOT = Path(__file__).resolve().parents[2] / "fpn_trainer"
_FPN_ROOM_ROOT = Path(__file__).resolve().parents[2] / "fpn_room_trainer"
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


def _load_fpn_model(root: Path, checkpoint: Path) -> tuple[Any, torch.device]:
    saved = {name: sys.modules.get(name) for name in _ISOLATED}
    root_str = str(root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    try:
        sys.modules["config"] = _load_module(f"{root.name}.config", root / "config.py")
        model_mod = _load_module(f"{root.name}.net.model", root / "net" / "model.py")
        cfg = sys.modules["config"].load_config()
        device = pick_device()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"FPN checkpoint not found: {checkpoint}")
        model = model_mod.build_model(cfg)
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        model.to(device)
        model.eval()
        return model, device
    finally:
        for name, module in saved.items():
            if module is not None:
                sys.modules[name] = module
            else:
                sys.modules.pop(name, None)


@dataclass
class FpnRuntime:
    structure_model: Any
    room_model: Any
    device: torch.device

    @classmethod
    def load(
        cls,
        *,
        structure_checkpoint: Path,
        room_checkpoint: Path,
    ) -> FpnRuntime:
        structure_model, device = _load_fpn_model(_FPN_ROOT, structure_checkpoint)
        room_model, room_device = _load_fpn_model(_FPN_ROOM_ROOT, room_checkpoint)
        if room_device != device:
            room_model.to(device)
        return cls(structure_model=structure_model, room_model=room_model, device=device)

    @torch.inference_mode()
    def infer_structure(self, prepared: Image.Image) -> np.ndarray:
        batch = image_to_tensor(prepared).unsqueeze(0).to(self.device)
        logits = self.structure_model(batch).mask_logits
        return torch.sigmoid(logits[0]).cpu().numpy()

    @torch.inference_mode()
    def infer_room(self, prepared: Image.Image) -> np.ndarray:
        batch = image_to_tensor(prepared).unsqueeze(0).to(self.device)
        logits = self.room_model(batch).mask_logits
        return torch.sigmoid(logits[0]).cpu().numpy()
