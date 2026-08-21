"""Run a LoRA Florence-2 adapter and parse room polygons from the generated text."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from config import Config
from data import (
    canvas_point_to_loc,
    format_infer_prompt,
    format_plot_text,
    polygon_to_canvas_ring,
    prepare_image,
    room_overlaps_existing,
    sample_random_canvas_points,
)
from loc import parse_referring_room
from model import load_model_and_processor


def _move_inputs_to_model(inputs: dict[str, Any], model: Any) -> dict[str, Any]:
    param = next(model.parameters())
    device, dtype = param.device, param.dtype
    return {
        key: (
            value.to(device=device, dtype=dtype)
            if torch.is_floating_point(value)
            else value.to(device)
        )
        if hasattr(value, "to")
        else value
        for key, value in inputs.items()
    }


def infer_image(
    model: Any,
    processor: Any,
    cfg: Config,
    image: Image.Image,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    try:
        image = prepare_image(image, cfg.image_size)
        rng = random.Random(cfg.infer.seed)
        points = sample_random_canvas_points(rng, cfg.canvas_size, cfg.infer.num_points)
        prompts = [
            format_infer_prompt(
                cfg.task_prompt,
                *canvas_point_to_loc(
                    x,
                    y,
                    canvas_size=cfg.canvas_size,
                    bins=cfg.loc_bins,
                ),
            )
            for x, y in points
        ]

        inputs = processor(
            text=prompts,
            images=[image] * len(prompts),
            return_tensors="pt",
            padding=True,
        )
        inputs = _move_inputs_to_model(inputs, model)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=cfg.infer.max_new_tokens,
                num_beams=cfg.infer.num_beams,
                do_sample=False,
                use_cache=True,
            )
        texts = processor.batch_decode(generated, skip_special_tokens=False)

        rooms: list[dict[str, Any]] = []
        canvas_rings: list[list[tuple[float, float]]] = []
        raw_outputs: list[str] = []
        for prompt, text in zip(prompts, texts, strict=True):
            if text.startswith(prompt):
                text = text[len(prompt) :]
            raw_outputs.append(text)
            room = parse_referring_room(text, image_size=cfg.image_size, bins=cfg.loc_bins)
            if room is None or len(room.get("polygon") or []) < 3:
                continue
            if room_overlaps_existing(
                room,
                canvas_rings,
                image_size=cfg.image_size,
                canvas_size=cfg.canvas_size,
            ):
                continue
            rooms.append(room)
            canvas_rings.append(
                polygon_to_canvas_ring(
                    room["polygon"],
                    image_size=cfg.image_size,
                    canvas_size=cfg.canvas_size,
                )
            )

        plot_text = format_plot_text(
            rooms,
            image_size=cfg.image_size,
            canvas_size=cfg.canvas_size,
            bins=cfg.loc_bins,
        )
        return {
            "rooms": rooms,
            "text": plot_text,
            "image": image,
            "prompts": prompts,
            "raw_outputs": raw_outputs,
        }
    finally:
        if was_training:
            model.train()


def run(
    cfg: Config,
    image_path: Path,
    *,
    adapter: Path | None = None,
    out: Path | None = None,
) -> dict[str, Any]:
    adapter_path = adapter or cfg.train.session_dir
    model, processor = load_model_and_processor(cfg, adapter=adapter_path)
    if torch.cuda.is_available():
        model.to("cuda")
    elif torch.backends.mps.is_available():
        model.to("mps")
    payload = infer_image(model, processor, cfg, Image.open(image_path))
    encoded = json.dumps(
        {
            "rooms": payload["rooms"],
            "text": payload["text"],
            "prompts": payload["prompts"],
        },
        indent=2,
    )
    if out is not None:
        out.write_text(encoded, encoding="utf-8")
    else:
        print(encoded)
    return {
        "rooms": payload["rooms"],
        "text": payload["text"],
        "prompts": payload["prompts"],
    }
