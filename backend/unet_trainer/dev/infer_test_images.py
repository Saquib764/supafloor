#!/usr/bin/env python3
"""Run test_image/ through TAESD encode → latent ViT → TAESD decode.

Usage (from backend/unet_trainer):

    uv run python dev/infer_test_images.py
    uv run python dev/infer_test_images.py --checkpoint training_session/last.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

CANVAS_SIZE = 1024
MARGIN = 10

_DEV = Path(__file__).resolve().parent
_ROOT = _DEV.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import CONFIG_PATH, load_config  # noqa: E402
from data import build_taesd, decode_latent, encode_image  # noqa: E402
from model import build_model  # noqa: E402

_DEFAULT_INPUT = _ROOT / "test_image"
_DEFAULT_OUTPUT = _DEV / "output" / "test-image-infer"
_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
_CHECKPOINT_NAME = "last.pt"


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def fit_to_canvas(
    image: Image.Image, size: int = CANVAS_SIZE, margin: int = MARGIN
) -> Image.Image:
    image = image.convert("L")
    fit_size = size - 2 * margin
    orig_w, orig_h = image.size
    scale = fit_size / max(orig_w, orig_h)
    new_w = max(1, round(orig_w * scale))
    new_h = max(1, round(orig_h * scale))
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (size, size), 255)
    canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
    return canvas


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.clip(np.asarray(image, dtype=np.float32) / 255.0, 0.0, 1.0)
    return torch.from_numpy(array).unsqueeze(0)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    array = (tensor.detach().float().clamp(0.0, 1.0).cpu().squeeze(0).numpy() * 255.0).astype(
        np.uint8
    )
    return Image.fromarray(array, mode="L")


def save_pair(input_pil: Image.Image, output_pil: Image.Image, path: Path) -> None:
    width, height = input_pil.size
    canvas = Image.new("L", (width * 2, height), 255)
    canvas.paste(input_pil, (0, 0))
    canvas.paste(output_pil, (width, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="JPEG", quality=95)


def infer_one(
    *,
    path: Path,
    out_dir: Path,
    taesd,
    model,
    device: torch.device,
) -> Path:
    input_pil = fit_to_canvas(Image.open(path), CANVAS_SIZE)
    x = pil_to_tensor(input_pil).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        z = encode_image(taesd, x)
        z_out = model(z)
        recon = decode_latent(taesd, z_out)

    output_pil = tensor_to_pil(recon[0])
    out_path = out_dir / f"{path.stem}_infer.jpg"
    save_pair(input_pil, output_pil, out_path)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Infer test_image/ through latent ViT")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=_ROOT / "training_session" / _CHECKPOINT_NAME,
        help="Trainable weights checkpoint (default: training_session/last.pt)",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=_DEFAULT_INPUT,
        help=f"Input image folder (default: {_DEFAULT_INPUT.relative_to(_ROOT)})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"Output folder (default: {_DEFAULT_OUTPUT.relative_to(_ROOT)})",
    )
    args = parser.parse_args(argv)

    input_dir = args.input if args.input.is_absolute() else _ROOT / args.input
    out_dir = args.output if args.output.is_absolute() else _ROOT / args.output
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else _ROOT / args.checkpoint

    paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in _EXTENSIONS)
    if not paths:
        raise SystemExit(f"No images found in {input_dir}")
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")

    cfg = load_config(args.config)
    device = pick_device()
    taesd = build_taesd(cfg.taesd.encoder, cfg.taesd.decoder, device)
    model = build_model(cfg).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_trainable_state_dict(state["trainable"])
    step = int(state.get("step", 0))

    print(f"Device: {device}")
    print(f"Canvas: {CANVAS_SIZE}×{CANVAS_SIZE}")
    print(f"Checkpoint: {checkpoint.relative_to(_ROOT)} (step {step})")

    for path in paths:
        out_path = infer_one(path=path, out_dir=out_dir, taesd=taesd, model=model, device=device)
        print(f"{path.name} → {out_path.relative_to(_ROOT)}")

    print(f"\nWrote {len(paths)} side-by-side panels (input | output) to {out_dir.relative_to(_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
