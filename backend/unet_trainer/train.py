"""Train the latent ViT: Hub image → simple_image (latent or image space)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader

from config import Config, datasets_cache_dir, load_config
from data import FloorplanPairDataset, build_taesd, decode_latent, hub_row_count, load_rows
from model import FloorplanLatentViT, image_recon_loss, latent_recon_loss
from preview import print_model

CHECKPOINT_NAME = "last.pt"


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def use_bf16(cfg: Config) -> bool:
    return bool(cfg.train.bf16 and torch.cuda.is_available())


def _tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    array = (tensor.detach().float().clamp(0.0, 1.0).cpu().squeeze(0).numpy() * 255.0).astype(
        np.uint8
    )
    return Image.fromarray(array, mode="L")


def _merge_recon_panels(
    source_image: torch.Tensor,
    target_image: torch.Tensor,
    recon_from_image: torch.Tensor,
    recon_from_simple: torch.Tensor,
    out_dir: Path,
    step: int,
) -> None:
    rows = (
        (
            _tensor_to_image(source_image[0]),
            _tensor_to_image(recon_from_image[0]),
            _tensor_to_image(target_image[0]),
        ),
        (
            _tensor_to_image(target_image[0]),
            _tensor_to_image(recon_from_simple[0]),
            _tensor_to_image(target_image[0]),
        ),
    )
    width, height = rows[0][0].size
    merged = Image.new("L", (width * 3, height * 2))
    for row_index, panels in enumerate(rows):
        for col_index, panel in enumerate(panels):
            merged.paste(panel, (col_index * width, row_index * height))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"step-{step:06d}.jpg"
    merged.save(path, format="JPEG", quality=95)
    print(f"recon → {path}")


def _save_recon_latent(
    model: FloorplanLatentViT,
    taesd,
    source_latent: torch.Tensor,
    target_latent: torch.Tensor,
    source_image: torch.Tensor,
    target_image: torch.Tensor,
    out_dir: Path,
    step: int,
    device: torch.device,
) -> None:
    model.eval()
    with torch.no_grad():
        pred_from_image = model(source_latent[:1].to(device))
        pred_from_simple = model(target_latent[:1].to(device))
        recon_from_image = decode_latent(taesd, pred_from_image)
        recon_from_simple = decode_latent(taesd, pred_from_simple)
    _merge_recon_panels(
        source_image, target_image, recon_from_image, recon_from_simple, out_dir, step
    )
    model.train()


def _save_recon_image(
    model: FloorplanLatentViT,
    source_image: torch.Tensor,
    target_image: torch.Tensor,
    out_dir: Path,
    step: int,
    device: torch.device,
) -> None:
    model.eval()
    with torch.no_grad():
        recon_from_image = model(source_image[:1].to(device))
        recon_from_simple = model(target_image[:1].to(device))
    _merge_recon_panels(
        source_image, target_image, recon_from_image, recon_from_simple, out_dir, step
    )
    model.train()


def _save_checkpoint(
    path: Path,
    *,
    model: FloorplanLatentViT,
    optimizer: AdamW,
    step: int,
    epoch: int,
) -> None:
    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "trainable": model.trainable_state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def _print_setup(cfg: Config, rows_len: int, steps_per_epoch: int, device: torch.device) -> None:
    precompute = cfg.train.precompute_latent
    print("\n=== Device ===")
    print(f"device:         {device}")
    print(f"cuda available: {torch.cuda.is_available()}")
    mps = getattr(torch.backends, "mps", None)
    print(f"mps available:  {mps is not None and mps.is_available()}")
    print(f"compute dtype:  {'bfloat16' if use_bf16(cfg) else 'float32'}")

    print("\n=== Dataset ===")
    print(f"dataset:        {cfg.data.dataset}")
    print(f"revision:       {cfg.data.revision or '(default)'}")
    print(f"cache_dir:      {datasets_cache_dir(cfg)}")
    print(f"hub rows:       {rows_len // 2:,}")
    print(f"train pairs:    {rows_len:,} (image→simple + simple→simple)")
    print(f"image_size:     {cfg.image_size}")
    if precompute:
        print(f"latent space:   taef2 ({cfg.image_size // 8}×{cfg.image_size // 8}), precomputed")
    else:
        print(f"latent space:   taef2 in model, image-space loss")

    print("\n=== Training ===")
    print(f"precompute_latent: {precompute}")
    print(f"output_dir:     {cfg.train.session_dir}")
    print(f"epochs:         {cfg.train.epochs}")
    print(f"batch_size:     {cfg.train.batch_size}")
    print(f"grad_accum:     {cfg.train.grad_accum}")
    print(f"effective batch:{cfg.train.batch_size * cfg.train.grad_accum}")
    print(f"steps/epoch:    {steps_per_epoch:,}")
    print(f"learning_rate:  {cfg.train.learning_rate}")
    print(f"ink_weight:     {cfg.loss.ink_weight}")
    print(f"resume:         {cfg.train.resume}")
    print(f"save_steps:     {cfg.train.save_steps}")
    print(f"recon_steps:    {cfg.train.recon_steps}")
    print()


def _train_latent(
    cfg: Config,
    *,
    model: FloorplanLatentViT,
    taesd,
    loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    recon_dir: Path,
    checkpoint_path: Path,
    amp: bool,
    start_epoch: int,
    step: int,
) -> int:
    accum = cfg.train.grad_accum
    optimizer.zero_grad(set_to_none=True)
    micro = 0
    for epoch in range(start_epoch, cfg.train.epochs):
        loader.dataset.epoch = epoch
        for source_latent, target_latent, weight, source_image, target_image in loader:
            source_latent = source_latent.to(device, non_blocking=True)
            target_latent = target_latent.to(device, non_blocking=True)
            weight = weight.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                pred = model(source_latent)
                loss = latent_recon_loss(pred, target_latent, weight=weight)
            (loss / accum).backward()
            micro += 1
            if micro % accum != 0:
                continue
            torch.nn.utils.clip_grad_norm_(
                (param for param in model.parameters() if param.requires_grad),
                cfg.train.max_grad_norm,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            print(f"step {step}  loss={loss.item():.4f}")
            if step % cfg.train.recon_steps == 0:
                _save_recon_latent(
                    model,
                    taesd,
                    source_latent,
                    target_latent,
                    source_image,
                    target_image,
                    recon_dir,
                    step,
                    device,
                )
            if step % cfg.train.save_steps == 0:
                _save_checkpoint(
                    checkpoint_path, model=model, optimizer=optimizer, step=step, epoch=epoch
                )
                print(f"saved → {checkpoint_path}")

    if micro % accum != 0:
        torch.nn.utils.clip_grad_norm_(
            (param for param in model.parameters() if param.requires_grad),
            cfg.train.max_grad_norm,
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
    return step


def _train_image(
    cfg: Config,
    *,
    model: FloorplanLatentViT,
    loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    recon_dir: Path,
    checkpoint_path: Path,
    amp: bool,
    start_epoch: int,
    step: int,
) -> int:
    accum = cfg.train.grad_accum
    optimizer.zero_grad(set_to_none=True)
    micro = 0
    for epoch in range(start_epoch, cfg.train.epochs):
        loader.dataset.epoch = epoch
        for source_image, target_image in loader:
            source_image = source_image.to(device, non_blocking=True)
            target_image = target_image.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                recon = model(source_image)
                loss = image_recon_loss(recon, target_image, ink_weight=cfg.loss.ink_weight)
            (loss / accum).backward()
            micro += 1
            if micro % accum != 0:
                continue
            torch.nn.utils.clip_grad_norm_(
                (param for param in model.parameters() if param.requires_grad),
                cfg.train.max_grad_norm,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            print(f"step {step}  loss={loss.item():.4f}")
            if step % cfg.train.recon_steps == 0:
                _save_recon_image(model, source_image, target_image, recon_dir, step, device)
            if step % cfg.train.save_steps == 0:
                _save_checkpoint(
                    checkpoint_path, model=model, optimizer=optimizer, step=step, epoch=epoch
                )
                print(f"saved → {checkpoint_path}")

    if micro % accum != 0:
        torch.nn.utils.clip_grad_norm_(
            (param for param in model.parameters() if param.requires_grad),
            cfg.train.max_grad_norm,
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
    return step


def run(cfg: Config) -> None:
    device = _device()
    precompute = cfg.train.precompute_latent
    taesd = build_taesd(cfg.taesd.encoder, cfg.taesd.decoder, device) if precompute else None
    rows = load_rows(
        cfg.data.dataset,
        revision=cfg.data.revision,
        cache_dir=datasets_cache_dir(cfg),
    )
    num_rows = hub_row_count(rows)
    dataset = FloorplanPairDataset(
        rows,
        cfg.image_size,
        num_rows=num_rows,
        precompute_latent=precompute,
        taesd=taesd,
        device=device,
        ink_weight=cfg.loss.ink_weight,
        seed=cfg.data.seed,
    )
    if num_rows == 0:
        raise FileNotFoundError(f"No rows in dataset {cfg.data.dataset!r}")
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    micro_per_epoch = max(1, (len(dataset) + cfg.train.batch_size - 1) // cfg.train.batch_size)
    steps_per_epoch = max(1, (micro_per_epoch + cfg.train.grad_accum - 1) // cfg.train.grad_accum)

    print("\n=== Model ===")
    model = print_model(cfg, dummy_forward=False)
    model.to(device)
    model.train()
    optimizer = AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )

    step = 0
    start_epoch = 0
    checkpoint_path = cfg.train.session_dir / CHECKPOINT_NAME
    if cfg.train.resume and checkpoint_path.is_file():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_trainable_state_dict(state["trainable"])
        optimizer.load_state_dict(state["optimizer"])
        step = int(state["step"])
        start_epoch = int(state["epoch"])
        print(f"Resumed step {step} epoch {start_epoch} ← {checkpoint_path}")

    _print_setup(cfg, len(dataset), steps_per_epoch, device)
    cfg.train.session_dir.mkdir(parents=True, exist_ok=True)
    recon_dir = cfg.train.session_dir / "recon"
    amp = use_bf16(cfg)

    if precompute:
        step = _train_latent(
            cfg,
            model=model,
            taesd=taesd,
            loader=loader,
            optimizer=optimizer,
            device=device,
            recon_dir=recon_dir,
            checkpoint_path=checkpoint_path,
            amp=amp,
            start_epoch=start_epoch,
            step=step,
        )
    else:
        step = _train_image(
            cfg,
            model=model,
            loader=loader,
            optimizer=optimizer,
            device=device,
            recon_dir=recon_dir,
            checkpoint_path=checkpoint_path,
            amp=amp,
            start_epoch=start_epoch,
            step=step,
        )

    _save_checkpoint(
        checkpoint_path, model=model, optimizer=optimizer, step=step, epoch=cfg.train.epochs
    )
    print(f"Saved → {checkpoint_path}")
    print(f"train={len(dataset)} steps={step}")


if __name__ == "__main__":
    run(load_config())
