"""Run a LoRA Florence-2 adapter and parse room polygons from the generated text."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from config import Config
from data import prepare_image
from loc import parse_rooms
from model import load_model_and_processor


def _device(model: Any) -> torch.device:
    return next(model.parameters()).device


def run(
    cfg: Config,
    image_path: Path,
    *,
    adapter: Path | None = None,
    out: Path | None = None,
) -> dict[str, Any]:
    adapter_path = adapter or cfg.train.output_dir
    model, processor = load_model_and_processor(cfg, adapter=adapter_path)
    if torch.cuda.is_available():
        model.to("cuda")
    elif torch.backends.mps.is_available():
        model.to("mps")
    model.eval()
    image = prepare_image(Image.open(image_path), cfg.image_size)
    inputs = processor(
        text=cfg.task_prompt,
        images=image,
        return_tensors="pt",
    )
    device = _device(model)
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=cfg.infer.max_new_tokens,
            num_beams=cfg.infer.num_beams,
            do_sample=False,
            use_cache=True,
        )
    text = processor.batch_decode(generated, skip_special_tokens=False)[0]
    if text.startswith(cfg.task_prompt):
        text = text[len(cfg.task_prompt) :]
    rooms = parse_rooms(text, image_size=cfg.image_size, bins=cfg.loc_bins)
    payload = {"rooms": rooms, "text": text}
    encoded = json.dumps(payload, indent=2)
    if out is not None:
        out.write_text(encoded, encoding="utf-8")
    else:
        print(encoded)
    return payload
