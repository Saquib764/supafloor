"""Print model/LoRA stats and random Hub dataset samples."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import torch

from config import CONFIG_PATH, Config, datasets_cache_dir, load_config

_PREVIEW_DIR = Path(__file__).resolve().parent / "_preview"


def _fmt_params(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    if n >= 1_000:
        return f"{n / 1e3:.2f}K"
    return str(n)


def _fmt_mb(num_bytes: int) -> str:
    return f"{num_bytes / (1024 ** 2):.2f} MB"


def _parameter_counts(model: Any) -> dict[str, int]:
    system_total = 0
    system_trainable = 0
    system_bytes = 0
    base_total = 0
    base_trainable = 0
    base_bytes = 0
    lora_total = 0
    lora_trainable = 0
    lora_bytes = 0

    for name, param in model.named_parameters():
        n = param.numel()
        nbytes = n * param.element_size()
        trainable = param.requires_grad
        system_total += n
        system_bytes += nbytes
        if trainable:
            system_trainable += n
        if "lora_" in name:
            lora_total += n
            lora_bytes += nbytes
            if trainable:
                lora_trainable += n
        else:
            base_total += n
            base_bytes += nbytes
            if trainable:
                base_trainable += n

    return {
        "system_total": system_total,
        "system_trainable": system_trainable,
        "system_bytes": system_bytes,
        "base_total": base_total,
        "base_trainable": base_trainable,
        "base_bytes": base_bytes,
        "lora_total": lora_total,
        "lora_trainable": lora_trainable,
        "lora_bytes": lora_bytes,
    }


def _print_parameter_counts(model: Any) -> None:
    counts = _parameter_counts(model)
    pct = 100 * counts["system_trainable"] / counts["system_total"] if counts["system_total"] else 0.0
    base_pct = (
        100 * counts["base_trainable"] / counts["base_total"] if counts["base_total"] else 0.0
    )

    print("\n=== Parameters (full system) ===")
    print(f"total:     {counts['system_total']:,} ({_fmt_params(counts['system_total'])})")
    print(f"size:      {_fmt_mb(counts['system_bytes'])}")
    print(
        f"trainable: {counts['system_trainable']:,} ({_fmt_params(counts['system_trainable'])}, "
        f"{pct:.4f}%)"
    )

    print("\n=== Parameters (base model) ===")
    print(f"total:     {counts['base_total']:,} ({_fmt_params(counts['base_total'])})")
    print(f"size:      {_fmt_mb(counts['base_bytes'])}")
    print(
        f"trainable: {counts['base_trainable']:,} ({_fmt_params(counts['base_trainable'])}, "
        f"{base_pct:.4f}%)"
    )

    print("\n=== Parameters (LoRA adapters) ===")
    print(f"total:     {counts['lora_total']:,} ({_fmt_params(counts['lora_total'])})")
    print(f"size:      {_fmt_mb(counts['lora_bytes'])}")
    print(f"trainable: {counts['lora_trainable']:,} ({_fmt_params(counts['lora_trainable'])})")


def _print_model_and_lora(model: Any, processor: Any, cfg: Config) -> None:
    print("=== Model ===")
    print(f"model_id:     {cfg.model_id}")
    print(f"class:        {model.__class__.__name__}")
    print(f"processor:    {processor.__class__.__name__}")
    print(f"dtype:        {next(model.parameters()).dtype}")
    print("\n=== LoRA ===")
    print(f"dropout:       {cfg.lora.dropout}")
    print(f"vision_rank:   {cfg.lora.vision_rank} (alpha {2 * cfg.lora.vision_rank})")
    print(f"encoder_rank:  {cfg.lora.encoder_rank} (alpha {2 * cfg.lora.encoder_rank})")
    print(f"decoder_rank:  {cfg.lora.decoder_rank} (alpha {2 * cfg.lora.decoder_rank})")
    _print_parameter_counts(model)


def _print_samples(cfg: Config, *, count: int, seed: int | None) -> None:
    import json

    from data import build_referring_variations, load_sample_rows, prepare_image

    cache_dir = datasets_cache_dir(cfg)
    print(f"Loading dataset into {cache_dir} …")
    samples, indices, num_rows = load_sample_rows(
        cfg.data.dataset,
        count=count,
        seed=seed,
        revision=cfg.data.revision,
        cache_dir=cache_dir,
    )

    if _PREVIEW_DIR.exists():
        shutil.rmtree(_PREVIEW_DIR)
    _PREVIEW_DIR.mkdir(parents=True)

    print("\n=== Dataset samples ===")
    print(f"dataset:      {cfg.data.dataset}")
    print(f"revision:     {cfg.data.revision or '(default)'}")
    print(f"rows:         {num_rows:,}")
    print(f"cache_dir:    {cache_dir}")
    print(f"preview_dir:  {_PREVIEW_DIR}")

    for n, (index, row) in enumerate(zip(indices, samples, strict=True), start=1):
        stem = str(row["stem"])
        image = prepare_image(row["image"], cfg.image_size)
        saved_path = _PREVIEW_DIR / f"{stem}.png"
        image.save(saved_path)
        extraction = json.loads(str(row["extraction"]))
        variations = build_referring_variations(
            extraction,
            index,
            canvas_size=cfg.canvas_size,
            bins=cfg.loc_bins,
            seed=cfg.data.seed,
            empty_polygon_prob=cfg.data.empty_polygon_prob,
            excluded_point_prob=cfg.data.excluded_point_prob,
            variations_per_floorplan=cfg.data.variations_per_floorplan,
        )

        print(f"\n--- sample {n} (index {index}) ---")
        print(f"stem:         {stem}")
        print(f"plan_id:      {row['plan_id']}")
        print(f"variant:      {row['variant']}")
        print(f"image_saved:  {saved_path}")
        print(f"image_size:   {image.size[0]}x{image.size[1]}")
        for v, datum in enumerate(variations, start=1):
            print(f"\n  variation {v}:")
            print(f"  prompt:           {datum.prompt!r}")
            print(f"  positive points:  {len(datum.positive_points)}")
            print(f"  excluded points:  {len(datum.excluded_points)}")
            print("  label:")
            text = datum.label if datum.label.endswith("\n") else f"{datum.label}\n"
            for line in text.splitlines():
                print(f"    {line}")


def run(cfg: Config, *, count: int = 3, seed: int | None = None) -> None:
    _print_samples(cfg, count=count, seed=seed)
    from model import load_model_and_processor

    print("\nLoading model with training LoRA attached …")
    model, processor = load_model_and_processor(cfg, for_training=True)
    model.eval()
    _print_model_and_lora(model, processor, cfg)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Print Florence-2 model, LoRA, and Hub samples")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--count", type=int, default=1, help="Random samples to print (default: 1)")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for sample selection")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    torch.set_grad_enabled(False)
    run(cfg, count=args.count, seed=args.seed)


if __name__ == "__main__":
    main()
