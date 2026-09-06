"""Print VAE architecture, parameter counts, and dummy-forward shapes."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config import CONFIG_PATH, Config, load_config
from model import FloorplanVAE

_ENCODER_ROOTS = frozenset({"stem", "downs", "to_mu", "to_logvar"})
_DECODER_ROOTS = frozenset({"from_z", "ups", "to_image"})


def _fmt_params(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    if n >= 1_000:
        return f"{n / 1e3:.2f}K"
    return str(n)


def _fmt_mb(num_bytes: int) -> str:
    return f"{num_bytes / (1024 ** 2):.2f} MB"


def _fmt_line(n: int, num_bytes: int) -> str:
    return f"{n:,} ({_fmt_params(n)}, {_fmt_mb(num_bytes)})"


def print_model(cfg: Config, *, dummy_forward: bool = True) -> FloorplanVAE:
    model = FloorplanVAE(
        image_size=cfg.image_size,
        patch_size=cfg.patch_size,
        in_channels=1,
        latent_channels=cfg.latent_channels,
    )
    model.eval()

    encoder = decoder = 0
    encoder_bytes = decoder_bytes = 0
    for name, param in model.named_parameters():
        n = param.numel()
        nbytes = n * param.element_size()
        root = name.split(".", 1)[0]
        if root in _ENCODER_ROOTS:
            encoder += n
            encoder_bytes += nbytes
        elif root in _DECODER_ROOTS:
            decoder += n
            decoder_bytes += nbytes
        else:
            raise RuntimeError(f"parameter not in encoder or decoder: {name}")

    print(model, flush=True)
    print("\n=== Parameters ===", flush=True)
    print(f"encoder: {_fmt_line(encoder, encoder_bytes)}", flush=True)
    print(f"decoder: {_fmt_line(decoder, decoder_bytes)}", flush=True)
    print(f"total:   {_fmt_line(encoder + decoder, encoder_bytes + decoder_bytes)}", flush=True)

    if dummy_forward:
        x = torch.zeros(1, 1, cfg.image_size, cfg.image_size)
        with torch.no_grad():
            recon, mu, logvar = model(x)
        latent = (cfg.latent_channels, cfg.latent_size, cfg.latent_size)
        if tuple(mu.shape[1:]) != latent or tuple(logvar.shape[1:]) != latent:
            raise RuntimeError(
                f"latent shape {tuple(mu.shape)} != (1, {latent[0]}, {latent[1]}, {latent[2]})"
            )
        if tuple(recon.shape) != tuple(x.shape):
            raise RuntimeError(f"recon shape {tuple(recon.shape)} != {tuple(x.shape)}")

        print("\n=== Dummy forward ===")
        print(f"x:      {tuple(x.shape)}")
        print(f"μ:      {tuple(mu.shape)}")
        print(f"logvar: {tuple(logvar.shape)}")
        print(f"x̂:      {tuple(recon.shape)}")

    print("\n=== Config ===")
    print(f"image_size:       {cfg.image_size}")
    print(f"patch_size:       {cfg.patch_size}")
    print(f"latent:           {cfg.latent_channels}×{cfg.latent_size}×{cfg.latent_size}")
    print(f"ink_weight:       {cfg.loss.ink_weight}")
    print(f"kl_beta:          {cfg.loss.kl_beta}")
    print(f"kl_warmup_steps:  {cfg.loss.kl_warmup_steps}")
    print(flush=True)
    return model


def run(cfg: Config) -> None:
    torch.set_grad_enabled(False)
    print_model(cfg)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Print floor-plan VAE architecture")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args(argv)
    run(load_config(args.config))


if __name__ == "__main__":
    main()
