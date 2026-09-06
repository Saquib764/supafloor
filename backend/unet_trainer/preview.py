"""Print latent-ViT architecture, parameter counts, and dummy-forward shapes."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config import CONFIG_PATH, Config, load_config
from data import taesd_hw
from model import FloorplanLatentViT, build_model

_TAESD_CHANNELS = 32
_FROZEN_ROOT = "taesd"
_TRAINABLE_ROOTS = ("compressor", "vit", "expander")


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


def print_model(cfg: Config, *, dummy_forward: bool = True) -> FloorplanLatentViT:
    model = build_model(cfg)
    model.eval()
    precompute = cfg.train.precompute_latent

    frozen = trainable = 0
    frozen_bytes = trainable_bytes = 0
    parts = {name: (0, 0) for name in _TRAINABLE_ROOTS}
    for name, param in model.named_parameters():
        n = param.numel()
        nbytes = n * param.element_size()
        root = name.split(".", 1)[0]
        if not precompute and root == _FROZEN_ROOT:
            frozen += n
            frozen_bytes += nbytes
        elif root in parts:
            count, size = parts[root]
            parts[root] = (count + n, size + nbytes)
            trainable += n
            trainable_bytes += nbytes
        else:
            raise RuntimeError(f"parameter not in taesd/compressor/vit/expander: {name}")

    print(model, flush=True)
    print("\n=== Parameters ===", flush=True)
    if not precompute:
        print(f"taesd (frozen): {_fmt_line(frozen, frozen_bytes)}", flush=True)
    for name in _TRAINABLE_ROOTS:
        count, size = parts[name]
        print(f"{name + ':':<16} {_fmt_line(count, size)}", flush=True)
    print(f"{'trainable:':<16} {_fmt_line(trainable, trainable_bytes)}", flush=True)
    if not precompute:
        print(f"{'total:':<16} {_fmt_line(frozen + trainable, frozen_bytes + trainable_bytes)}", flush=True)

    if dummy_forward:
        with torch.no_grad():
            if precompute:
                spatial = taesd_hw(cfg.image_size)
                z = torch.zeros(1, _TAESD_CHANNELS, spatial, spatial)
                z_comp = model.compressor(z)
                z_vit = model.vit(z_comp)
                z_up = model.expander(z_vit)
                if tuple(z_up.shape) != tuple(z.shape):
                    raise RuntimeError(f"output shape {tuple(z_up.shape)} != {tuple(z.shape)}")
                print("\n=== Dummy forward (latent) ===")
                print(f"z_in:        {tuple(z.shape)}")
                print(f"z_compress:  {tuple(z_comp.shape)}")
                print(f"z_vit:       {tuple(z_vit.shape)}")
                print(f"z_out:       {tuple(z_up.shape)}")
            else:
                x = torch.zeros(1, 1, cfg.image_size, cfg.image_size)
                z_taesd = model.encode_taesd(x)
                z_comp = model.compressor(z_taesd)
                z_vit = model.vit(z_comp)
                z_up = model.expander(z_vit)
                recon = model.decode_taesd(z_up)
                if tuple(recon.shape) != tuple(x.shape):
                    raise RuntimeError(f"recon shape {tuple(recon.shape)} != {tuple(x.shape)}")
                print("\n=== Dummy forward (image) ===")
                print(f"x:           {tuple(x.shape)}")
                print(f"z_taesd:     {tuple(z_taesd.shape)}")
                print(f"z_compress:  {tuple(z_comp.shape)}")
                print(f"z_vit:       {tuple(z_vit.shape)}")
                print(f"z_up:        {tuple(z_up.shape)}")
                print(f"x̂:           {tuple(recon.shape)}")

    print("\n=== Config ===")
    print(f"image_size:       {cfg.image_size}")
    print(f"precompute_latent:{precompute}")
    print(f"taef2 latent:     {_TAESD_CHANNELS}×{taesd_hw(cfg.image_size)}×{taesd_hw(cfg.image_size)}")
    print(f"compressed:       16×{cfg.compressed_size}×{cfg.compressed_size}")
    print(f"vit.dim:          {cfg.vit.dim}")
    print(f"vit.depth:        {cfg.vit.depth}")
    print(f"vit.heads:        {cfg.vit.heads}")
    print(f"vit.patch_size:   {cfg.vit.patch_size}")
    print(f"ink_weight:       {cfg.loss.ink_weight}")
    print(flush=True)
    return model


def run(cfg: Config) -> None:
    torch.set_grad_enabled(False)
    print_model(cfg)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Print floor-plan latent ViT architecture")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args(argv)
    run(load_config(args.config))


if __name__ == "__main__":
    main()
