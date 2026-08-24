"""Training-time test inference: sample random canvas points and decode room polygons."""

from __future__ import annotations

import random
import time
from typing import Any

from PIL import Image

from shapely.geometry import Polygon

from data import (
    format_plot_text,
    format_prompt,
    polygon_to_canvas_ring,
    prepare_image,
    room_overlaps_existing,
    sample_points,
)
from config import Config
from inference.generate import generate_for_prompts
from inference.params import infer_params_from_config
from inference.progressive import MAX_POINTS_PER_PROMPT
from loc import parse_rooms


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
        params = infer_params_from_config(cfg)
        image = prepare_image(image, params.image_size)
        rng = random.Random(params.seed)
        _, points = sample_points(
            Polygon(),
            rng,
            canvas_size=params.canvas_size,
            inside_count=0,
            outside_count=cfg.infer.num_points,
        )
        batches = [
            points[i : i + MAX_POINTS_PER_PROMPT]
            for i in range(0, len(points), MAX_POINTS_PER_PROMPT)
        ]
        prompts = [
            format_prompt(
                batch,
                canvas_size=params.canvas_size,
                bins=params.loc_bins,
            )
            for batch in batches
        ]

        texts = generate_for_prompts(model, processor, params, image, prompts)

        rooms: list[dict[str, Any]] = []
        canvas_rings: list[list[tuple[float, float]]] = []
        raw_outputs: list[str] = []
        point_scale = params.point_scale
        for batch, prompt, text in zip(batches, prompts, texts, strict=True):
            if text.startswith(prompt):
                text = text[len(prompt) :]
            raw_outputs.append(text)
            parsed = parse_rooms(text, image_size=params.image_size, bins=params.loc_bins)
            for index, (x, y) in enumerate(batch):
                room = parsed[index] if index < len(parsed) else None
                if room is None or len(room.get("polygon") or []) < 3:
                    continue
                if room_overlaps_existing(
                    room,
                    canvas_rings,
                    image_size=params.image_size,
                    canvas_size=params.canvas_size,
                ):
                    continue
                room["sample_point"] = [x * point_scale, y * point_scale]
                rooms.append(room)
                canvas_rings.append(
                    polygon_to_canvas_ring(
                        room["polygon"],
                        image_size=params.image_size,
                        canvas_size=params.canvas_size,
                    )
                )

        plot_text = format_plot_text(
            rooms,
            image_size=params.image_size,
            canvas_size=params.canvas_size,
            bins=params.loc_bins,
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
