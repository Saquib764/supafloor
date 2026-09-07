"""Run the latest unet_trainer checkpoint through taef2 encode/decode."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

_UNET_ROOT = Path(__file__).resolve().parents[2] / "unet_trainer"
_CHECKPOINT_NAME = "last.pt"
_ISOLATED = ("config", "data", "taesd", "model")


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
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


@dataclass
class UnetRuntime:
    image_size: int
    device: torch.device
    taesd: Any
    model: Any
    prepare_image: Any
    encode_image: Any
    decode_latent: Any

    @classmethod
    def load(cls, checkpoint: Path | None = None) -> UnetRuntime:
        saved = {name: sys.modules.get(name) for name in _ISOLATED}
        unet_str = str(_UNET_ROOT.resolve())
        if unet_str not in sys.path:
            sys.path.insert(0, unet_str)
        try:
            sys.modules["config"] = _load_module("unet_trainer.config", _UNET_ROOT / "config.py")
            sys.modules["taesd"] = _load_module("unet_trainer.taesd", _UNET_ROOT / "taesd.py")
            sys.modules["data"] = _load_module("unet_trainer.data", _UNET_ROOT / "data.py")
            sys.modules["model"] = _load_module("unet_trainer.model", _UNET_ROOT / "model.py")

            unet_config = sys.modules["config"]
            unet_data = sys.modules["data"]
            unet_model = sys.modules["model"]
            cfg = unet_config.load_config()
            device = pick_device()
            checkpoint = checkpoint or (_UNET_ROOT / cfg.train.session_dir / _CHECKPOINT_NAME)
            if not checkpoint.is_file():
                raise FileNotFoundError(f"UNet checkpoint not found: {checkpoint}")

            taesd = unet_data.build_taesd(cfg.taesd.encoder, cfg.taesd.decoder, device)
            model = unet_model.build_model(cfg).to(device)
            state = torch.load(checkpoint, map_location=device, weights_only=False)
            model.load_trainable_state_dict(state["trainable"])
            model.eval()
            return cls(
                image_size=int(cfg.image_size),
                device=device,
                taesd=taesd,
                model=model,
                prepare_image=unet_data.prepare_image,
                encode_image=unet_data.encode_image,
                decode_latent=unet_data.decode_latent,
            )
        finally:
            for name, module in saved.items():
                if module is not None:
                    sys.modules[name] = module
                else:
                    sys.modules.pop(name, None)


    def infer(self, image: Image.Image) -> Image.Image:
        x = self.prepare_image(image.convert("L"), self.image_size).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            z = self.encode_image(self.taesd, x)
            z_out = self.model(z)
            recon = self.decode_latent(self.taesd, z_out)
        array = (
            recon[0, 0].detach().float().clamp(0.0, 1.0).cpu().numpy() * 255.0
        ).astype(np.uint8)
        return Image.fromarray(array, mode="L").convert("RGB")


@lru_cache(maxsize=1)
def get_unet_runtime(checkpoint: str) -> UnetRuntime:
    path = Path(checkpoint) if checkpoint else None
    return UnetRuntime.load(path)
