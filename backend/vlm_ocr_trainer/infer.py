"""Run OCR inference on images from test_images/."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from benchmark import load_model_from_checkpoint
from renderer.canvas import input_output_row
from config import CONFIG_PATH, Config, load_config
from data import image_to_tensor
from device import training_device
from model import OcrVlm

_TEST_IMAGES_DIR = CONFIG_PATH.parent / "test_images"
_CANVAS_SIZE = 256
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def fit_to_canvas(image: Image.Image, *, size: int = _CANVAS_SIZE) -> Image.Image:
    """Scale to fit a square canvas (aspect preserved), center on white background."""
    if image.mode != "L":
        image = image.convert("L")
    orig_w, orig_h = image.size
    scale = size / max(orig_w, orig_h)
    new_w = max(1, round(orig_w * scale))
    new_h = max(1, round(orig_h * scale))
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (size, size), 255)
    canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
    return canvas


def load_image(path: Path, *, canvas_size: int = _CANVAS_SIZE) -> Image.Image:
    return fit_to_canvas(Image.open(path), size=canvas_size)


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
        generated = model.generate(tensor, max_length=cfg.max_seq_len.final)
    return model.vocab.decode(generated[0].tolist())


def _list_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"input directory not found: {directory}")
    paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    )
    if not paths:
        raise FileNotFoundError(f"no images found in {directory}")
    return paths


def run(
    cfg: Config,
    *,
    checkpoint: Path | None = None,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    canvas_size: int = _CANVAS_SIZE,
) -> list[tuple[Path, str]]:
    device = training_device()
    checkpoint_path = checkpoint or cfg.infer.checkpoint
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    model = load_model_from_checkpoint(cfg, checkpoint_path, device)
    image_dir = input_dir or _TEST_IMAGES_DIR
    out_dir = output_dir or (cfg.train.session_dir / "infer")
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[tuple[Path, str]] = []
    for path in _list_images(image_dir):
        prepared = load_image(path, canvas_size=canvas_size)
        text = infer_text(model, prepared, cfg=cfg, device=device)
        results.append((path, text))

        prepared.save(out_dir / f"{path.stem}-input.jpg", format="JPEG", quality=95)
        panel = input_output_row(prepared, text)
        panel.save(out_dir / f"{path.stem}-preview.jpg", format="JPEG", quality=95)

        print(f"{path.name}: {text!r}")
        print(f"  input   → {out_dir / f'{path.stem}-input.jpg'}")
        print(f"  preview → {out_dir / f'{path.stem}-preview.jpg'}")

    print(f"\ncheckpoint: {checkpoint_path}")
    print(f"output_dir: {out_dir}")
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run OCR inference on test images")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path (default: infer.checkpoint from config)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=_TEST_IMAGES_DIR,
        help=f"Directory of input images (default: {_TEST_IMAGES_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for prepared inputs and previews (default: training_session/infer)",
    )
    parser.add_argument(
        "--canvas-size",
        type=int,
        default=_CANVAS_SIZE,
        help="Square white canvas size in pixels (default: 256)",
    )
    args = parser.parse_args(argv)
    run(
        load_config(args.config),
        checkpoint=args.checkpoint,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        canvas_size=args.canvas_size,
    )


if __name__ == "__main__":
    main()
