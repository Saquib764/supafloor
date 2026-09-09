"""Run the latest fpn_trainer checkpoint to detect walls, doors, and windows."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps

_FPN_ROOT = Path(__file__).resolve().parents[2] / "fpn_trainer"
_INFER_SIZE = 1024
_ISOLATED = ("config",)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _prepare_image(image: Image.Image, size: int) -> Image.Image:
    gray = image.convert("L")
    if gray.size != (size, size):
        gray = gray.resize((size, size), Image.Resampling.LANCZOS)
    return ImageOps.invert(gray)


def _image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def _masks_to_rgb(masks: np.ndarray) -> np.ndarray:
    """(C, H, W) bool/float → (H, W, 3) float. R=walls, G=doors, B=windows."""
    active = masks > 0.5
    height, width = active.shape[-2:]
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    for channel in range(min(active.shape[0], 3)):
        rgb[..., channel] = active[channel] * 255.0
    return rgb


def overlay_structure(base: Image.Image, masks: np.ndarray) -> Image.Image:
    base_rgb = base.convert("RGB")
    mask_rgb = _masks_to_rgb(masks)
    mask_pil = Image.fromarray(mask_rgb.astype(np.uint8)).resize(
        base_rgb.size,
        Image.Resampling.NEAREST,
    )
    base_arr = np.array(base_rgb, dtype=np.float32)
    mask_arr = np.array(mask_pil, dtype=np.float32)
    active = mask_arr.max(axis=-1) > 127
    out = base_arr.copy()
    out[active] = base_arr[active] * 0.45 + mask_arr[active] * 0.55
    return Image.fromarray(out.astype(np.uint8))


@dataclass
class FpnRuntime:
    device: torch.device
    model: Any

    @classmethod
    def load(cls, checkpoint: Path | None = None) -> FpnRuntime:
        saved = {name: sys.modules.get(name) for name in _ISOLATED}
        fpn_str = str(_FPN_ROOT.resolve())
        if fpn_str not in sys.path:
            sys.path.insert(0, fpn_str)
        try:
            sys.modules["config"] = _load_module("fpn_trainer.config", _FPN_ROOT / "config.py")
            fpn_model = _load_module("fpn_trainer.net.model", _FPN_ROOT / "net" / "model.py")

            fpn_config = sys.modules["config"]
            cfg = fpn_config.load_config()
            device = pick_device()
            checkpoint = checkpoint or (_FPN_ROOT / cfg.infer.checkpoint)
            if not checkpoint.is_file():
                raise FileNotFoundError(f"FPN checkpoint not found: {checkpoint}")

            model = fpn_model.build_model(cfg)
            state = torch.load(checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(state["model"])
            model.to(device)
            model.eval()
            return cls(device=device, model=model)
        finally:
            for name, module in saved.items():
                if module is not None:
                    sys.modules[name] = module
                else:
                    sys.modules.pop(name, None)

    def infer_structure(self, image: Image.Image) -> Image.Image:
        original_size = image.size
        size = _INFER_SIZE
        small = image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)

        prepared = _prepare_image(small, size)
        batch = _image_to_tensor(prepared).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(batch).mask_logits
        mask = (torch.sigmoid(logits[0]) > 0.5).cpu().numpy()

        output = overlay_structure(small, mask)
        if output.size != original_size:
            output = output.resize(original_size, Image.Resampling.LANCZOS)
        return output


@lru_cache(maxsize=1)
def get_fpn_runtime(checkpoint: str) -> FpnRuntime:
    path = Path(checkpoint) if checkpoint else None
    return FpnRuntime.load(path)
