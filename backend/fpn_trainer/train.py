"""Train star-prompted room segmentation (CNN + FPN)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader

from canvas import merge_panels, overlay_masks
from config import Config, datasets_cache_dir, load_config
from data import (
    FloorplanStarDataset,
    StarCollator,
    build_star_samples,
    extraction_scale,
    image_to_tensor,
    load_train_split,
    load_viz_rows,
    prepare_image,
    scale_extraction,
)
from net.loss import multi_mask_loss
from net.model import StarRoomFPN
from preview import print_model
from star import draw_filled_star

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


def _save_checkpoint(
    path: Path,
    *,
    model: StarRoomFPN,
    optimizer: AdamW,
    step: int,
    epoch: int,
) -> None:
    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def _viz_row(
    model: StarRoomFPN,
    row: dict[str, object],
    cfg: Config,
    device: torch.device,
    *,
    index: int,
    step: int,
) -> Image.Image:
    base = prepare_image(row["image"], cfg.input_size)
    extraction = scale_extraction(
        json.loads(str(row["extraction"])),
        extraction_scale(cfg.input_size),
    )
    samples = build_star_samples(extraction, index + step * 9973, cfg)
    if not samples:
        raise ValueError("no star samples for viz row")
    rng = random.Random((cfg.data.seed or 0) + index + step)
    (cx, cy, radius), targets = samples[rng.randrange(len(samples))]
    starred = base.copy()
    draw_filled_star(
        starred,
        cx,
        cy,
        radius,
        color=cfg.star.color,
        inner_ratio=cfg.star.inner_ratio,
    )
    batch = image_to_tensor(starred).unsqueeze(0)
    batch = batch.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(batch).mask_logits
    pred = (torch.sigmoid(logits[0]) > 0.5).cpu().numpy()
    model.train()
    return merge_panels(
        [
            starred,
            overlay_masks(starred, pred),
            overlay_masks(starred, targets),
        ]
    )


def _save_viz(
    model: StarRoomFPN,
    cfg: Config,
    device: torch.device,
    train_count: int,
    out_dir: Path,
    step: int,
) -> None:
    rows = load_viz_rows(
        cfg.data.dataset,
        train_count,
        count=cfg.train.viz_samples,
        revision=cfg.data.revision,
        cache_dir=datasets_cache_dir(cfg),
    )
    if not rows:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows):
        panel = _viz_row(model, row, cfg, device, index=index, step=step)
        path = out_dir / f"step-{step:06d}-{row['stem']}.jpg"
        panel.save(path, format="JPEG", quality=95)
        print(f"viz → {path}")


def _print_setup(cfg: Config, dataset_len: int, steps_per_epoch: int, device: torch.device) -> None:
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
    print(f"train samples:  {dataset_len:,}")
    print(f"input_size:     {cfg.input_size}")

    print("\n=== Training ===")
    print(f"output_dir:     {cfg.train.session_dir}")
    print(f"epochs:         {cfg.train.epochs}")
    print(f"batch_size:     {cfg.train.batch_size}")
    print(f"grad_accum:     {cfg.train.grad_accum}")
    print(f"steps/epoch:    {steps_per_epoch:,}")
    print(f"learning_rate:  {cfg.train.learning_rate}")
    print(f"resume:         {cfg.train.resume}")
    print(f"save_steps:     {cfg.train.save_steps}")
    print(f"viz_steps:      {cfg.train.viz_steps}")
    print()


def run(cfg: Config) -> None:
    device = _device()
    train_rows, train_count, _test_count = load_train_split(
        cfg.data.dataset,
        test_fraction=cfg.data.test_fraction,
        revision=cfg.data.revision,
        cache_dir=datasets_cache_dir(cfg),
    )
    dataset = FloorplanStarDataset(train_rows, cfg, num_rows=train_count)
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=StarCollator(),
        drop_last=False,
    )
    micro_per_epoch = max(1, (len(dataset) + cfg.train.batch_size - 1) // cfg.train.batch_size)
    steps_per_epoch = max(1, (micro_per_epoch + cfg.train.grad_accum - 1) // cfg.train.grad_accum)

    print("\n=== Model ===")
    model = print_model(cfg, dummy_forward=False)
    model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)

    step = 0
    start_epoch = 0
    checkpoint_path = cfg.train.session_dir / CHECKPOINT_NAME
    if cfg.train.resume and checkpoint_path.is_file():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step = int(state["step"])
        start_epoch = int(state["epoch"])
        print(f"Resumed step {step} epoch {start_epoch} ← {checkpoint_path}")

    _print_setup(cfg, len(dataset), steps_per_epoch, device)
    cfg.train.session_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = cfg.train.session_dir / "viz"
    amp = use_bf16(cfg)
    accum = cfg.train.grad_accum
    optimizer.zero_grad(set_to_none=True)
    micro = 0

    for epoch in range(start_epoch, cfg.train.epochs):
        dataset.epoch = epoch
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            masks = batch["masks"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits = model(images).mask_logits
                loss = multi_mask_loss(
                    logits,
                    masks,
                    focal_alpha=cfg.loss.focal_alpha,
                    focal_gamma=cfg.loss.focal_gamma,
                    channel_weights=cfg.loss.channel_weights,
                )
            (loss / accum).backward()
            micro += 1
            if micro % accum != 0:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            print(f"step {step}  loss={loss.item():.4f}")
            if step % cfg.train.viz_steps == 0:
                _save_viz(model, cfg, device, train_count, viz_dir, step)
            if step % cfg.train.save_steps == 0:
                _save_checkpoint(
                    checkpoint_path, model=model, optimizer=optimizer, step=step, epoch=epoch
                )
                print(f"saved → {checkpoint_path}")

    if micro % accum != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

    _save_checkpoint(
        checkpoint_path, model=model, optimizer=optimizer, step=step, epoch=cfg.train.epochs
    )
    print(f"Saved → {checkpoint_path}")
    print(f"train={len(dataset)} steps={step}")


if __name__ == "__main__":
    run(load_config())
