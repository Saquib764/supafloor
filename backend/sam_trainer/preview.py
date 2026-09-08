"""Print SAM 2.1 architecture and save random training samples."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from canvas import apply_masks
from config import CONFIG_PATH, Config, datasets_cache_dir, load_config
from data import build_room_samples, load_sample_rows, prepare_image
from model import load_predictor, lora_param_count, pick_device, trainable_param_count

_PREVIEW_DIR = CONFIG_PATH.parent / "_preview"
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
    print(f"lora rank:          {cfg.lora.rank}")
    print(f"lora stage3:        {cfg.lora.stage3}")
    print(f"lora stage4:        {cfg.lora.stage4}")
    if cfg.infer.checkpoint is not None:
        print(f"infer checkpoint:   {cfg.infer.checkpoint}")
    print(f"lora trainable:     {lora_param_count(model):,}")
    print(f"finetune trainable: {trainable_param_count(model):,}")
    print(flush=True)
    return model


def save_data_samples(cfg: Config, *, count: int, seed: int | None) -> None:
    cache_dir = datasets_cache_dir(cfg)
    print(f"Loading dataset into {cache_dir} …")
    rows, indices, num_rows = load_sample_rows(
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

    for n, (index, row) in enumerate(zip(indices, rows, strict=True), start=1):
        stem = str(row["stem"])
        image = prepare_image(row["image"], cfg.canvas_size)
        image_path = _PREVIEW_DIR / f"{stem}.jpg"
        image.save(image_path, format="JPEG", quality=95)

        extraction = json.loads(str(row["extraction"]))
        samples = build_room_samples(
            extraction,
            index,
            canvas_size=cfg.canvas_size,
            room_slots=cfg.data.room_slots,
            per_room_variation=cfg.data.per_room_variation,
            point_distance_alpha=cfg.data.point_distance_alpha,
            seed=cfg.data.seed,
        )

        print(f"\n--- sample {n} (index {index}) ---")
        print(f"stem:         {stem}")
        print(f"image:        {image_path}")
        for v, (point, mask) in enumerate(samples, start=1):
            gt_path = _PREVIEW_DIR / f"{stem}_v{v}_gt.jpg"
            apply_masks(image, [mask.astype(bool)], [point]).save(gt_path, format="JPEG", quality=95)
            print(f"  variation {v}: point=({point[0]:.1f},{point[1]:.1f}) gt={gt_path.name}")


def run(cfg: Config, *, count: int = 1, seed: int | None = None) -> None:
    torch.set_grad_enabled(False)
    if count > 0:
        save_data_samples(cfg, count=count, seed=seed)
    print("\nLoading model …")
    print_model(cfg)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Print SAM 2.1 architecture and save Hub samples")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--count", type=int, default=1, help="Random samples to save (0 = model only)")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for sample selection")
    args = parser.parse_args(argv)
    run(load_config(args.config), count=args.count, seed=args.seed)


if __name__ == "__main__":
    main()
