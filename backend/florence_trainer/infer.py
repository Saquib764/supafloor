"""Training-time test inference: sample random canvas points and decode room polygons."""

from __future__ import annotations

import random
import time
from typing import Any

import torch
from PIL import Image

from config import Config
from data import (
    format_plot_text,
    format_training_prompt,
    polygon_to_canvas_ring,
    prepare_image,
    room_overlaps_existing,
    sample_random_canvas_points,
)
from loc import parse_rooms

MAX_POINTS_PER_PROMPT = 6


def _generate_for_prompts(
    model: Any,
    processor: Any,
    cfg: Config,
    image: Image.Image,
    prompts: list[str],
) -> list[str]:
    gen_kwargs = {
        "max_new_tokens": cfg.infer.max_new_tokens,
        "num_beams": cfg.infer.num_beams,
        "do_sample": False,
        "use_cache": True,
    }
    if torch.cuda.is_available():
        inputs = processor(
            text=prompts,
            images=[image] * len(prompts),
            return_tensors="pt",
            padding=True,
        )
        inputs = _move_inputs_to_model(inputs, model)
        with torch.inference_mode():
            generated = model.generate(**inputs, **gen_kwargs)
        return processor.batch_decode(generated, skip_special_tokens=False)

    texts: list[str] = []
    for prompt in prompts:
        inputs = processor(text=[prompt], images=[image], return_tensors="pt")
        inputs = _move_inputs_to_model(inputs, model)
        with torch.inference_mode():
            generated = model.generate(**inputs, **gen_kwargs)
        texts.extend(processor.batch_decode(generated, skip_special_tokens=False))
    return texts


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
    started = time.perf_counter()
    try:
        image = prepare_image(image, cfg.image_size)
        rng = random.Random(cfg.infer.seed)
        points = sample_random_canvas_points(rng, cfg.canvas_size, cfg.infer.num_points)
        batches = [
            points[i : i + MAX_POINTS_PER_PROMPT]
            for i in range(0, len(points), MAX_POINTS_PER_PROMPT)
        ]
        prompts = [
            format_training_prompt(
                cfg.task_prompt,
                batch,
                canvas_size=cfg.canvas_size,
                bins=cfg.loc_bins,
            )
            for batch in batches
        ]

        texts = _generate_for_prompts(model, processor, cfg, image, prompts)

        rooms: list[dict[str, Any]] = []
        canvas_rings: list[list[tuple[float, float]]] = []
        raw_outputs: list[str] = []
        point_scale = cfg.image_size / cfg.canvas_size
        for batch, prompt, text in zip(batches, prompts, texts, strict=True):
            if text.startswith(prompt):
                text = text[len(prompt) :]
            raw_outputs.append(text)
            parsed = parse_rooms(text, image_size=cfg.image_size, bins=cfg.loc_bins)
            for index, (x, y) in enumerate(batch):
                room = parsed[index] if index < len(parsed) else None
                if room is None or len(room.get("polygon") or []) < 3:
                    continue
                if room_overlaps_existing(
                    room,
                    canvas_rings,
                    image_size=cfg.image_size,
                    canvas_size=cfg.canvas_size,
                ):
                    continue
                room["sample_point"] = [x * point_scale, y * point_scale]
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
        elapsed = time.perf_counter() - started
        print(f"Test infer: {len(points)} points in {elapsed:.2f}s ({len(rooms)} rooms)")
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
