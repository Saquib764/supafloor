"""Run star-prompted room segmentation on a single image."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from config import Config
from data import image_to_tensor, prepare_image
from net.model import StarRoomFPN, build_model
from star import draw_filled_star


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(cfg: Config, checkpoint: Path, device: torch.device) -> StarRoomFPN:
    model = build_model(cfg)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()
    return model


def infer_mask(
    model: StarRoomFPN,
    image: Image.Image,
    *,
    cx: float,
    cy: float,
    radius: float,
    cfg: Config,
    device: torch.device,
) -> tuple[Image.Image, np.ndarray]:
    base = prepare_image(image, cfg.input_size)
    scale_x = cfg.input_size / image.size[0]
    scale_y = cfg.input_size / image.size[1]
    sx = cx * scale_x
    sy = cy * scale_y
    starred = base.copy()
    draw_filled_star(
        starred,
        sx,
        sy,
        radius,
        color=cfg.star.color,
        inner_ratio=cfg.star.inner_ratio,
    )
    batch = image_to_tensor(starred).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(batch).mask_logits
    mask = (torch.sigmoid(logits[0]) > 0.5).cpu().numpy()
    return starred, mask
