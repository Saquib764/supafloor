"""Shared Florence model generation helpers."""

from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from inference.params import InferParams


def move_inputs_to_model(inputs: dict[str, Any], model: Any) -> dict[str, Any]:
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


def generate_for_prompts(
    model: Any,
    processor: Any,
    params: InferParams,
    image: Image.Image,
    prompts: list[str],
) -> list[str]:
    gen_kwargs = {
        "max_new_tokens": params.max_new_tokens,
        "num_beams": params.num_beams,
        "do_sample": False,
        "use_cache": True,
    }
    texts: list[str] = []
    for prompt in prompts:
        inputs = processor(text=[prompt], images=[image], return_tensors="pt")
        inputs = move_inputs_to_model(inputs, model)
        with torch.inference_mode():
            generated = model.generate(**inputs, **gen_kwargs)
        texts.extend(processor.batch_decode(generated, skip_special_tokens=False))
    return texts
