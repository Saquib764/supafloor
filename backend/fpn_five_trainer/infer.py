"""Run floor-plan structure segmentation on a single image."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from config import Config
from data import image_to_tensor, prepare_image
from net.model import FloorplanFPN, build_model


def load_model(cfg: Config, checkpoint: Path, device: torch.device) -> FloorplanFPN:
    model = build_model(cfg)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.to(device)
    model.eval()
    return model


def infer_mask(
    model: FloorplanFPN,
    image: Image.Image,
    *,
    cfg: Config,
    device: torch.device,
) -> tuple[Image.Image, np.ndarray]:
    prepared = prepare_image(image, cfg.infer.size)
    batch = image_to_tensor(prepared).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(batch).mask_logits
    mask = (torch.sigmoid(logits[0]) > 0.5).cpu().numpy()
    return prepared, mask
