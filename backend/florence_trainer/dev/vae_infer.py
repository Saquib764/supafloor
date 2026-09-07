"""Run the latest vae_trainer checkpoint (image → simple_image recon)."""

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

_VAE_ROOT = Path(__file__).resolve().parents[2] / "vae_trainer"
_CHECKPOINT_NAME = "last.pt"
_ISOLATED = ("config", "data", "model")


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
class VaeRuntime:
    image_size: int
    device: torch.device
    model: Any
    prepare_image: Any

    @classmethod
    def load(cls, checkpoint: Path | None = None) -> VaeRuntime:
        saved = {name: sys.modules.get(name) for name in _ISOLATED}
        vae_str = str(_VAE_ROOT.resolve())
        if vae_str not in sys.path:
            sys.path.insert(0, vae_str)
        try:
            sys.modules["config"] = _load_module("vae_trainer.config", _VAE_ROOT / "config.py")
            sys.modules["data"] = _load_module("vae_trainer.data", _VAE_ROOT / "data.py")
            sys.modules["model"] = _load_module("vae_trainer.model", _VAE_ROOT / "model.py")

            vae_config = sys.modules["config"]
            vae_data = sys.modules["data"]
            vae_model = sys.modules["model"]
            cfg = vae_config.load_config()
            device = pick_device()
            checkpoint = checkpoint or (_VAE_ROOT / cfg.train.session_dir / _CHECKPOINT_NAME)
            if not checkpoint.is_file():
                raise FileNotFoundError(f"VAE checkpoint not found: {checkpoint}")

            model = vae_model.FloorplanVAE(
                image_size=cfg.image_size,
                patch_size=cfg.patch_size,
                in_channels=1,
                latent_channels=cfg.latent_channels,
            ).to(device)
            state = torch.load(checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(state["model"])
            model.eval()
            return cls(
                image_size=int(cfg.image_size),
                device=device,
                model=model,
                prepare_image=vae_data.prepare_image,
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
            mu, _logvar = self.model.encode(x)
            recon = self.model.decode(mu)
        array = (
            recon[0, 0].detach().float().clamp(0.0, 1.0).cpu().numpy() * 255.0
        ).astype(np.uint8)
        return Image.fromarray(array, mode="L").convert("RGB")


@lru_cache(maxsize=1)
def get_vae_runtime(checkpoint: str) -> VaeRuntime:
    path = Path(checkpoint) if checkpoint else None
    return VaeRuntime.load(path)
