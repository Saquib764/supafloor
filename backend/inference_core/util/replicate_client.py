"""Replicate image editing."""

from __future__ import annotations

import base64
import io
import os

import replicate
from PIL import Image


def _image_to_data_uri(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=95)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _read_output_image(output: object) -> Image.Image:
    if hasattr(output, "read"):
        data = output.read()
        if isinstance(data, str):
            data = data.encode("utf-8")
        return Image.open(io.BytesIO(data)).convert("RGB")
    if isinstance(output, (bytes, bytearray)):
        return Image.open(io.BytesIO(output)).convert("RGB")
    if isinstance(output, str):
        if output.startswith("http"):
            import urllib.request

            with urllib.request.urlopen(output) as response:
                return Image.open(io.BytesIO(response.read())).convert("RGB")
        return Image.open(output).convert("RGB")
    if isinstance(output, list) and output:
        return _read_output_image(output[0])
    raise ValueError(f"unexpected Replicate output type: {type(output)!r}")


def run_replicate_edit(
    image: Image.Image,
    prompt: str,
    *,
    model: str,
    token: str,
    quality: str,
    aspect_ratio: str,
    output_format: str,
    background: str,
    moderation: str,
    number_of_images: int,
) -> Image.Image:
    os.environ["REPLICATE_API_TOKEN"] = token
    output = replicate.run(
        model,
        input={
            "prompt": prompt,
            "input_images": [_image_to_data_uri(image)],
            "quality": quality,
            "aspect_ratio": aspect_ratio,
            "output_format": output_format,
            "background": background,
            "moderation": moderation,
            "number_of_images": number_of_images,
        },
    )
    return _read_output_image(output)
