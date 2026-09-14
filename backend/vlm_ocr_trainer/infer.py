"""Run OCR inference on text samples rendered from test_samples/samples.json."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from benchmark import load_model_from_checkpoint
from renderer.canvas import input_output_row
from config import CONFIG_PATH, Config, VariationSettings, load_config
from data import image_to_tensor
from device import training_device
from model import OcrVlm, validate_image_size
from renderer.paragraph_render import (
    render_paragraph_document,
    render_paragraph_from_variations,
)

_TEST_SAMPLES_DIR = CONFIG_PATH.parent / "test_samples"
_SAMPLES_JSON = _TEST_SAMPLES_DIR / "samples.json"
_DEFAULT_CANVAS_SIZE = 256
_MAX_SEQ_LEN = 256
_META_KEYS = frozenset({"width", "height", "background", "noise_prob"})


def _parse_canvas_size(meta: dict[str, Any]) -> tuple[int, int]:
    width = int(meta.get("width", _DEFAULT_CANVAS_SIZE))
    height = int(meta.get("height", _DEFAULT_CANVAS_SIZE))
    validate_image_size(width, height)
    return width, height


def fit_to_canvas(
    image: Image.Image,
    *,
    width: int,
    height: int,
) -> Image.Image:
    """Scale to fit a canvas (aspect preserved), center on white background."""
    if image.mode != "L":
        image = image.convert("L")
    if image.size == (width, height):
        return image
    orig_w, orig_h = image.size
    scale = min(width / orig_w, height / orig_h)
    new_w = max(1, round(orig_w * scale))
    new_h = max(1, round(orig_h * scale))
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (width, height), 255)
    canvas.paste(resized, ((width - new_w) // 2, (height - new_h) // 2))
    return canvas


def _variation_settings(base: VariationSettings, meta: dict[str, Any]) -> VariationSettings:
    updates: dict[str, Any] = {}
    if "background" in meta:
        bg = int(meta["background"])
        updates["bg_min"] = bg
        updates["bg_max"] = bg
    if "noise_prob" in meta:
        updates["noise_prob"] = float(meta["noise_prob"])
    return replace(base, **updates) if updates else base


def load_samples(
    path: Path,
) -> tuple[tuple[int, int], dict[str, Any], dict[str, dict[str, Any]], bool]:
    if not path.is_file():
        raise FileNotFoundError(f"samples file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{path} must be a non-empty object")
    meta = {key: raw[key] for key in _META_KEYS if key in raw}
    had_canvas_size = "width" in meta and "height" in meta
    canvas_size = _parse_canvas_size(meta)
    samples: dict[str, dict[str, Any]] = {}
    for sample_id, entry in raw.items():
        if sample_id in _META_KEYS:
            continue
        if isinstance(entry, str):
            samples[str(sample_id)] = {"text": entry}
            continue
        if not isinstance(entry, dict) or "text" not in entry:
            raise ValueError(f"{path}[{sample_id!r}] must be a string or {{\"text\": ...}}")
        samples[str(sample_id)] = dict(entry)
    if not samples:
        raise ValueError(f"{path} must include at least one sample")
    return canvas_size, meta, samples, had_canvas_size


def save_samples(
    path: Path,
    samples: dict[str, dict[str, Any]],
    *,
    meta: dict[str, Any],
    canvas_size: tuple[int, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas_w, canvas_h = canvas_size
    payload: dict[str, Any] = {
        key: value for key, value in meta.items() if key not in {"width", "height"}
    }
    payload["width"] = canvas_w
    payload["height"] = canvas_h
    for sample_id in sorted(samples):
        payload[sample_id] = samples[sample_id]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _paragraphs_from_text(text: str) -> list[str]:
    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    return paragraphs or [text]


def _sample_rng(entry: dict[str, Any], *, reseed: bool) -> tuple[random.Random, bool]:
    if not reseed and "seed" in entry:
        return random.Random(int(entry["seed"])), False
    seed = random.randint(0, 2**31 - 1)
    entry["seed"] = seed
    return random.Random(seed), True


@torch.inference_mode()
def infer_text(
    model: OcrVlm,
    image: Image.Image,
    *,
    cfg: Config,
    device: torch.device,
) -> str:
    tensor = image_to_tensor(image).unsqueeze(0).to(device)
    amp = bool(cfg.train.bf16 and device.type == "cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        generated = model.generate(tensor, max_length=_MAX_SEQ_LEN)
    return model.vocab.decode(generated[0].tolist())


def run(
    cfg: Config,
    *,
    checkpoint: Path | None = None,
    samples_path: Path | None = None,
    output_dir: Path | None = None,
    canvas_size: tuple[int, int] | None = None,
    reseed: bool = False,
) -> list[tuple[str, str]]:
    device = training_device()
    checkpoint_path = checkpoint or cfg.infer.checkpoint
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    print(f"checkpoint: {checkpoint_path.resolve()}")
    model = load_model_from_checkpoint(cfg, checkpoint_path, device)
    json_path = samples_path or _SAMPLES_JSON
    file_canvas_size, meta, samples, had_canvas_size = load_samples(json_path)
    canvas_w, canvas_h = canvas_size or file_canvas_size
    out_dir = output_dir or (cfg.train.session_dir / "infer")
    out_dir.mkdir(parents=True, exist_ok=True)

    variations_settings = _variation_settings(cfg.data.variations.final, meta)
    modified = canvas_size is not None and canvas_size != file_canvas_size
    results: list[tuple[str, str]] = []

    for sample_id in sorted(samples):
        entry = samples[sample_id]
        paragraphs = _paragraphs_from_text(str(entry["text"]))
        saved_variations = entry.get("variations")

        if saved_variations and not reseed:
            rendered = render_paragraph_from_variations(
                canvas_w,
                canvas_h,
                paragraphs,
                saved_variations,
            )
        else:
            sample_rng, seeded = _sample_rng(entry, reseed=reseed)
            if seeded:
                modified = True
            rendered = render_paragraph_document(
                canvas_w,
                canvas_h,
                sample_rng,
                paragraphs,
                variations_settings,
            )
            if rendered.variations != saved_variations:
                entry["variations"] = rendered.variations
                modified = True

        prepared = fit_to_canvas(rendered.image, width=canvas_w, height=canvas_h)
        text = infer_text(model, prepared, cfg=cfg, device=device)
        results.append((sample_id, text))

        panel = input_output_row(prepared, text)
        panel.save(out_dir / f"{sample_id}-preview.jpg", format="JPEG", quality=95)

        print(f"{sample_id}: {text!r}")
        print(f"  preview → {out_dir / f'{sample_id}-preview.jpg'}")

    save_canvas_size = (canvas_w, canvas_h)
    save_meta = dict(meta)
    if modified or not had_canvas_size or file_canvas_size != save_canvas_size:
        save_samples(
            json_path,
            samples,
            meta=save_meta,
            canvas_size=save_canvas_size,
        )

    print(f"\nsamples:    {json_path}")
    print(f"canvas:     {canvas_w}x{canvas_h}")
    print(f"output_dir: {out_dir}")
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run OCR inference on rendered test samples")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path (default: infer.checkpoint from config)",
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=_SAMPLES_JSON,
        help=f"Sample definitions JSON (default: {_SAMPLES_JSON})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for preview panels (default: training_session/infer)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Override canvas width in pixels (default: width from samples JSON)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Override canvas height in pixels (default: height from samples JSON)",
    )
    parser.add_argument(
        "--reseed",
        action="store_true",
        help="Ignore saved seeds and generate new random render variations",
    )
    args = parser.parse_args(argv)
    canvas_override: tuple[int, int] | None = None
    if args.width is not None or args.height is not None:
        canvas_override = (
            args.width if args.width is not None else _DEFAULT_CANVAS_SIZE,
            args.height if args.height is not None else _DEFAULT_CANVAS_SIZE,
        )
    run(
        load_config(args.config),
        checkpoint=args.checkpoint,
        samples_path=args.samples,
        output_dir=args.output_dir,
        canvas_size=canvas_override,
        reseed=args.reseed,
    )


if __name__ == "__main__":
    main()
