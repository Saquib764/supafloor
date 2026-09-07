"""Print SAM 2.1 architecture and parameter counts by module."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config import CONFIG_PATH, Config, load_config
from model import load_predictor, pick_device

_MODULES = ("vision_encoder", "prompt_encoder", "mask_decoder", "shared_image_embedding")


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


def _count_module(model: torch.nn.Module, prefix: str) -> tuple[int, int]:
    count = num_bytes = 0
    for name, param in model.named_parameters():
        if not name.startswith(prefix):
            continue
        n = param.numel()
        count += n
        num_bytes += n * param.element_size()
    return count, num_bytes


def print_model(cfg: Config) -> torch.nn.Module:
    device = pick_device()
    predictor = load_predictor(cfg)
    model = predictor.model
    model.eval()

    parts: dict[str, tuple[int, int]] = {}
    for name in _MODULES:
        parts[name] = _count_module(model, f"{name}.")

    total = sum(n for n, _ in parts.values())
    total_bytes = sum(b for _, b in parts.values())

    print(model, flush=True)
    print("\n=== Parameters ===", flush=True)
    for name in _MODULES:
        count, size = parts[name]
        if count:
            print(f"{name + ':':<24} {_fmt_line(count, size)}", flush=True)
    print(f"{'total:':<24} {_fmt_line(total, total_bytes)}", flush=True)

    for module_name in ("vision_encoder", "prompt_encoder", "mask_decoder"):
        module = getattr(model, module_name)
        subs = [(name, mod) for name, mod in module.named_children()]
        if not subs:
            continue
        print(f"\n=== {module_name} ===", flush=True)
        for name, mod in subs:
            n = sum(p.numel() for p in mod.parameters())
            if not n:
                continue
            b = sum(p.numel() * p.element_size() for p in mod.parameters())
            print(f"  {name + ':':<22} {_fmt_line(n, b)}", flush=True)

    print("\n=== Config ===")
    print(f"model_id:           {cfg.model_id}")
    print(f"device:             {device}")
    print(f"canvas_size:        {cfg.canvas_size}")
    print(f"multimask_output:   {cfg.infer.multimask_output}")
    print(f"hf_cache:           {cfg.hf_cache}")
    print(flush=True)
    return model


def run(cfg: Config) -> None:
    torch.set_grad_enabled(False)
    print_model(cfg)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Print SAM 2.1 architecture and parameter counts")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args(argv)
    run(load_config(args.config))


if __name__ == "__main__":
    main()
