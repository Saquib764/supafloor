"""Print Sana stack stats and optional sample generation."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from diffusers import SanaPipeline

from config import CONFIG_PATH, Config, load_config
from model import (
    count_parameters,
    load_training_stack,
    lora_parameter_stats,
    mixed_precision_dtype,
    unwrap_transformer,
)

_PREVIEW_DIR = CONFIG_PATH.parent / "_preview"


def _fmt_params(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    if n >= 1_000:
        return f"{n / 1e3:.2f}K"
    return str(n)


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def print_model(cfg: Config, stack) -> None:
    mode = "full transformer" if cfg.train.full_finetune else "LoRA"
    mp, dtype = mixed_precision_dtype(cfg)
    total, trainable, _ = count_parameters(stack.transformer)
    print("\n=== Sana training stack ===")
    print(f"model_id:       {cfg.model_id}")
    print(f"mode:           {mode}")
    print(f"weight_dtype:   {dtype}")
    print(f"mixed_precision:{mp}")
    print(f"transformer:    {total:,} params ({_fmt_params(total)})")
    print(f"trainable:      {trainable:,} params ({_fmt_params(trainable)})")
    if not cfg.train.full_finetune:
        lora_total, lora_train = lora_parameter_stats(stack.transformer)
        print("\n=== LoRA ===")
        print(f"rank:           {cfg.lora.rank}")
        print(f"alpha:          {cfg.lora.alpha}")
        print(f"dropout:        {cfg.lora.dropout}")
        print(f"target_modules: {', '.join(cfg.lora.target_modules)}")
        print(f"lora params:    {lora_total:,} ({_fmt_params(lora_total)})")
        print(f"lora trainable: {lora_train:,} ({_fmt_params(lora_train)})")


def _generate_preview(cfg: Config, stack, device: torch.device) -> None:
    if _PREVIEW_DIR.exists():
        shutil.rmtree(_PREVIEW_DIR)
    _PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    pipe = SanaPipeline.from_pretrained(
        cfg.model_id,
        transformer=unwrap_transformer(stack.transformer),
        text_encoder=stack.text_encoder,
        tokenizer=stack.tokenizer,
        vae=stack.vae,
        revision=cfg.revision,
        variant=cfg.variant,
        torch_dtype=stack.weight_dtype,
    )
    pipe = pipe.to(device)
    pipe.text_encoder.to(torch.bfloat16)
    pipe.vae.to(torch.float32)
    pipe.set_progress_bar_config(disable=True)
    gen = torch.Generator(device=device).manual_seed(cfg.preview.seed)
    image = pipe(
        prompt=cfg.preview.prompt,
        height=cfg.resolution,
        width=cfg.resolution,
        complex_human_instruction=cfg.complex_human_instruction,
        generator=gen,
        num_inference_steps=20,
    ).images[0]
    out = _PREVIEW_DIR / "preview.jpg"
    image.save(out, format="JPEG", quality=95)
    print(f"\npreview image:  {out}")


def run(cfg: Config, *, skip_image: bool = False) -> None:
    device = _device()
    adapter = cfg.train.from_adapter
    stack = load_training_stack(cfg, device=device, adapter=adapter)
    print_model(cfg, stack)
    print(f"\npreview prompt: {cfg.preview.prompt!r}")
    print(f"preview_dir:    {_PREVIEW_DIR}")
    if skip_image:
        return
    if device.type == "cpu":
        print("\n(skip generation: CUDA/MPS not available)")
        return
    _generate_preview(cfg, stack, device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview Sana model and LoRA setup")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--skip-image", action="store_true")
    args = parser.parse_args()
    run(load_config(args.config), skip_image=args.skip_image)


if __name__ == "__main__":
    main()
