#!/usr/bin/env python3
"""Encode/decode test floorplans with TAESD for FLUX.2 (taef2 weights).

Usage (from backend/vae_trainer):

    uv run python dev/flux2_roundtrip.py
    uv run python dev/flux2_roundtrip.py --input test_floorplans --output dev/output/flux2-roundtrip
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_DEV = Path(__file__).resolve().parent
_ROOT = _DEV.parent
if str(_DEV) not in sys.path:
    sys.path.insert(0, str(_DEV))

from taesd import TAESD  # noqa: E402

RESOLUTION = 512

_DEFAULT_INPUT = _ROOT / "test_floorplans"
_DEFAULT_OUTPUT = _DEV / "output" / "flux2-roundtrip"
_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
_WEIGHTS_DIR = _DEV / "weights"
_ENCODER_URL = "https://github.com/madebyollin/taesd/raw/main/taef2_encoder.pth"
_DECODER_URL = "https://github.com/madebyollin/taesd/raw/main/taef2_decoder.pth"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {dest.name} …")
    urllib.request.urlretrieve(url, dest)


def ensure_weights() -> tuple[Path, Path]:
    encoder = _WEIGHTS_DIR / "taef2_encoder.pth"
    decoder = _WEIGHTS_DIR / "taef2_decoder.pth"
    if not encoder.exists():
        _download(_ENCODER_URL, encoder)
    if not decoder.exists():
        _download(_DECODER_URL, decoder)
    return encoder, decoder


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(rgb).permute(2, 0, 1)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    arr = (tensor.detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def mse_psnr(original: torch.Tensor, recon: torch.Tensor) -> tuple[float, float]:
    mse = float(torch.mean((original - recon) ** 2))
    if mse == 0:
        return 0.0, float("inf")
    psnr = 10 * np.log10(1.0 / mse)
    return mse, float(psnr)


def save_triptych(original: Image.Image, recon: Image.Image, diff: Image.Image, path: Path) -> None:
    w, h = original.size
    canvas = Image.new("RGB", (w * 3, h), "white")
    canvas.paste(original, (0, 0))
    canvas.paste(recon, (w, 0))
    canvas.paste(diff, (w * 2, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=92)


def diff_image(original: torch.Tensor, recon: torch.Tensor) -> Image.Image:
    err = torch.abs(original - recon).mean(dim=0)
    err = (err / err.max().clamp(min=1e-6) * 255).round().byte().cpu().numpy()
    return Image.fromarray(err, mode="L").convert("RGB")


def load_resized(path: Path, size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.LANCZOS)
    return image


def roundtrip_one(
    taesd: TAESD, path: Path, out_dir: Path, *, resolution: int
) -> dict[str, float | str | tuple[int, int] | tuple[int, int, int]]:
    original_pil = load_resized(path, resolution)
    original = pil_to_tensor(original_pil).unsqueeze(0).to(next(taesd.parameters()).device)

    with torch.no_grad():
        latent = taesd.encoder(original)
        recon = taesd.decoder(latent).clamp(0, 1)

    mse, psnr = mse_psnr(original, recon)
    recon_pil = tensor_to_pil(recon[0])
    diff_pil = diff_image(original[0], recon[0])
    out_path = out_dir / f"{path.stem}_roundtrip.jpg"
    save_triptych(original_pil, recon_pil, diff_pil, out_path)

    _, lc, lh, lw = latent.shape
    return {
        "file": path.name,
        "input_size": original_pil.size,
        "latent_shape": (lc, lh, lw),
        "mse": mse,
        "psnr": psnr,
        "output": str(out_path.relative_to(_ROOT)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TAESD FLUX.2 roundtrip on test floorplans")
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

    paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in _EXTENSIONS)
    if not paths:
        raise SystemExit(f"No images found in {input_dir}")

    device = pick_device()
    encoder_path, decoder_path = ensure_weights()
    taesd = TAESD(encoder_path=str(encoder_path), decoder_path=str(decoder_path)).to(device)
    taesd.eval()
    print(f"Device: {device}")
    latent_ch = taesd.encoder[-1].out_channels
    print(f"TAESD FLUX.2 (8× downscale, {latent_ch} latent channels, {RESOLUTION}×{RESOLUTION} input)")

    rows: list[dict[str, float | str | tuple[int, int]]] = []
    for path in paths:
        row = roundtrip_one(taesd, path, out_dir, resolution=RESOLUTION)
        rows.append(row)
        print(
            f"{row['file']}: latent {row['latent_shape'][0]}×{row['latent_shape'][1]}×{row['latent_shape'][2]} "
            f"(C×H×W) mse={row['mse']:.6f} psnr={row['psnr']:.2f} dB → {row['output']}"
        )

    mse_avg = sum(float(r["mse"]) for r in rows) / len(rows)
    psnr_avg = sum(float(r["psnr"]) for r in rows) / len(rows)
    print(f"\n{len(rows)} images — avg mse={mse_avg:.6f}, avg psnr={psnr_avg:.2f} dB")
    print(f"Wrote triptychs (original | recon | |Δ|) to {out_dir.relative_to(_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
