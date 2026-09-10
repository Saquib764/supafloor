"""Run the latest fpn_room_trainer checkpoint to segment room interiors."""

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

_FPN_ROOM_ROOT = Path(__file__).resolve().parents[2] / "fpn_room_trainer"
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


def overlay_room_mask(base: Image.Image, mask: np.ndarray) -> Image.Image:
    """(C, H, W) bool/float → RGB overlay. Room interior shown in green."""
    if mask.ndim != 3 or mask.shape[0] < 1:
        raise ValueError(f"expected (1, H, W) mask, got {mask.shape}")
    active = mask[0] > 0.5
    base_rgb = base.convert("RGB")
    mask_rgb = np.zeros((active.shape[0], active.shape[1], 3), dtype=np.float32)
    mask_rgb[..., 1] = active * 255.0
    mask_pil = Image.fromarray(mask_rgb.astype(np.uint8)).resize(
        base_rgb.size,
        Image.Resampling.NEAREST,
    )
    base_arr = np.array(base_rgb, dtype=np.float32)
    mask_arr = np.array(mask_pil, dtype=np.float32)
    active_full = mask_arr[..., 1] > 127
    out = base_arr.copy()
    out[active_full] = base_arr[active_full] * 0.45 + mask_arr[active_full] * 0.55
    return Image.fromarray(out.astype(np.uint8))


@dataclass
class FpnRoomRuntime:
    device: torch.device
    model: Any
    infer_size: int

    @classmethod
    def load(cls, checkpoint: Path | None = None) -> FpnRoomRuntime:
        saved = {name: sys.modules.get(name) for name in _ISOLATED}
        root = str(_FPN_ROOM_ROOT.resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
        try:
            sys.modules["config"] = _load_module(
                "fpn_room_trainer.config",
                _FPN_ROOM_ROOT / "config.py",
            )
            fpn_model = _load_module(
                "fpn_room_trainer.net.model",
                _FPN_ROOM_ROOT / "net" / "model.py",
            )

            fpn_config = sys.modules["config"]
            cfg = fpn_config.load_config()
            device = pick_device()
            checkpoint = checkpoint or (_FPN_ROOM_ROOT / cfg.infer.checkpoint)
            if not checkpoint.is_file():
                raise FileNotFoundError(f"FPN room checkpoint not found: {checkpoint}")

            model = fpn_model.build_model(cfg)
            state = torch.load(checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(state["model"])
            model.to(device)
            model.eval()
            return cls(device=device, model=model, infer_size=cfg.infer.size)
        finally:
            for name, module in saved.items():
                if module is not None:
                    sys.modules[name] = module
                else:
                    sys.modules.pop(name, None)

    def infer_rooms(self, image: Image.Image) -> Image.Image:
        original_size = image.size
        size = self.infer_size
        small = image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)

        prepared = _prepare_image(small, size)
        batch = _image_to_tensor(prepared).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(batch).mask_logits
        mask = (torch.sigmoid(logits[0]) > 0.5).cpu().numpy()

        output = overlay_room_mask(small, mask)
        if output.size != original_size:
            output = output.resize(original_size, Image.Resampling.LANCZOS)
        return output


@lru_cache(maxsize=1)
def get_fpn_room_runtime(checkpoint: str | None = None) -> FpnRoomRuntime:
    path = Path(checkpoint) if checkpoint else None
    return FpnRoomRuntime.load(path)
