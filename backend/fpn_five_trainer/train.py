"""Train floor-plan structure segmentation (CNN + FPN)."""

from __future__ import annotations

import math
import random
import time
from pathlib import Path

import torch
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from canvas import columns_input_masks, stack_rows
from config import Config, datasets_cache_dir, load_config
from data import (
    FloorplanCollator,
    FloorplanDataset,
    augment_sample,
    image_to_tensor,
    load_train_split,
    load_training_sample,
    load_viz_rows,
)
from device import training_device
from net.loss import five_mask_loss
from net.model import FloorplanFPN
from preview import print_model
from registry import local_dataset_available, local_dataset_root

CHECKPOINT_NAME = "last.pt"


def use_bf16(cfg: Config) -> bool:
    return bool(cfg.train.bf16 and torch.cuda.is_available())


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        synchronize = getattr(torch.mps, "synchronize", None)
        if synchronize is not None:
            synchronize()


def _ema(prev: float, value: float, *, alpha: float = 0.1) -> float:
    return value if prev == 0.0 else alpha * value + (1.0 - alpha) * prev


def _lr_lambda(*, warmup_steps: int, total_steps: int):
    def fn(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return 1.0
        progress = float(step - warmup_steps) / float(total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return fn


def _save_checkpoint(
    path: Path,
    *,
    model: FloorplanFPN,
    optimizer: AdamW,
    scheduler: LambdaLR,
    step: int,
    epoch: int,
) -> None:
    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        path,
    )


def _viz_row(
    model: FloorplanFPN,
    row: dict[str, object],
    cfg: Config,
    device: torch.device,
    *,
    seed: int,
) -> Image.Image:
    size = max(cfg.input_sizes)
    image, targets = load_training_sample(
        row,
        dataset_name=cfg.data.dataset,
        size=size,
        mask_channels=cfg.mask_channels,
        image_key="image",
    )
    image, targets = augment_sample(
        image,
        targets,
        rng=random.Random(seed),
        augment=cfg.data.augment,
        background_channel=cfg.mask_channels.background,
        size=size,
    )
    batch = image_to_tensor(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(batch).mask_logits
    pred = (torch.sigmoid(logits[0]) > 0.5).cpu().numpy()
    model.train()
    labels = cfg.mask_channels.labels()
    pred_row = columns_input_masks(
        image, pred, mask_labels=labels, background_channel=cfg.mask_channels.background
    )
    gt_row = columns_input_masks(
        image, targets, mask_labels=labels, background_channel=cfg.mask_channels.background
    )
    return stack_rows([pred_row, gt_row])


def _save_viz(
    model: FloorplanFPN,
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
        source=cfg.data.source,
        seed=(cfg.data.seed or 0) + step,
        revision=cfg.data.revision,
        cache_dir=datasets_cache_dir(cfg),
    )
    if not rows:
        return
    step_dir = out_dir / f"step-{step:06d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows):
        panel = _viz_row(
            model,
            row,
            cfg,
            device,
            seed=(cfg.data.seed or 0) + step + index,
        )
        path = step_dir / f"{row['stem']}.jpg"
        panel.save(path, format="JPEG", quality=95)
    tqdm.write(f"viz → {step_dir} ({len(rows)} test image(s))")


def _print_setup(
    cfg: Config,
    dataset_len: int,
    steps_per_epoch: int,
    total_steps: int,
    warmup_steps: int,
    device: torch.device,
) -> None:
    print("\n=== Device ===")
    print(f"device:         {device}")
    print(f"cuda available: {torch.cuda.is_available()}")
    mps = getattr(torch.backends, "mps", None)
    print(f"mps available:  {mps is not None and mps.is_available()}")
    print(f"compute dtype:  {'bfloat16' if use_bf16(cfg) else 'float32'}")

    print("\n=== Dataset ===")
    print(f"dataset:        {cfg.data.dataset}")
    print(f"source:         {cfg.data.source}")
    local_root = local_dataset_root(cfg.data.dataset)
    if cfg.data.source in ("auto", "local") and local_root is not None:
        loaded_local = local_dataset_available(cfg.data.dataset)
        print(f"local_root:     {local_root} ({'loaded' if loaded_local else 'missing'})")
    print(f"revision:       {cfg.data.revision or '(default)'}")
    print(f"cache_dir:      {datasets_cache_dir(cfg)}")
    print(f"train rows:     {dataset_len:,}")
    print(f"input_sizes:    {list(cfg.input_sizes)}")

    print("\n=== Training ===")
    print(f"output_dir:     {cfg.train.session_dir}")
    print(f"init_checkpoint:{cfg.train.init_checkpoint or '(none)'}")
    print(f"epochs:         {cfg.train.epochs}")
    print(f"batch_size:     {cfg.train.batch_size}")
    print(f"grad_accum:     {cfg.train.grad_accum}")
    print(f"steps/epoch:    {steps_per_epoch:,}")
    print(f"total_steps:    {total_steps:,}")
    print(f"warmup_steps:   {warmup_steps:,}")
    print(f"learning_rate:  {cfg.train.learning_rate}")
    print(f"warmup_ratio:   {cfg.train.warmup_ratio}")
    print(f"resume:         {cfg.train.resume}")
    print(f"save_steps:     {cfg.train.save_steps}")
    print(f"viz_steps:      {cfg.train.viz_steps}")
    use_workers = cfg.data.num_workers > 0
    thread_prefetch = 0 if use_workers else cfg.data.prefetch_samples
    if thread_prefetch == 0 and not use_workers:
        thread_prefetch = cfg.train.batch_size * 4
    print(f"num_workers:    {cfg.data.num_workers}")
    print(
        f"prefetch:       factor={cfg.data.prefetch_factor}, "
        f"samples={thread_prefetch if not use_workers else 0}"
    )
    print()


def run(cfg: Config) -> None:
    device = training_device()
    train_rows, train_count, _test_count = load_train_split(
        cfg.data.dataset,
        test_fraction=cfg.data.test_fraction,
        source=cfg.data.source,
        revision=cfg.data.revision,
        cache_dir=datasets_cache_dir(cfg),
    )
    use_workers = cfg.data.num_workers > 0
    prefetch_samples = 0 if use_workers else cfg.data.prefetch_samples
    if prefetch_samples == 0 and not use_workers:
        prefetch_samples = cfg.train.batch_size * 4
    dataset = FloorplanDataset(
        train_rows,
        dataset_name=cfg.data.dataset,
        mask_channels=cfg.mask_channels,
        input_sizes=cfg.input_sizes,
        batch_size=cfg.train.batch_size,
        num_rows=train_count,
        augment=cfg.data.augment,
        cache_size=max(cfg.input_sizes),
        seed=cfg.data.seed,
        prefetch_samples=prefetch_samples,
    )
    loader_kwargs: dict[str, object] = {
        "batch_size": cfg.train.batch_size,
        "shuffle": False,
        "num_workers": cfg.data.num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": FloorplanCollator(),
        "drop_last": False,
    }
    if use_workers:
        loader_kwargs["prefetch_factor"] = cfg.data.prefetch_factor
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(dataset, **loader_kwargs)
    micro_per_epoch = max(1, (len(dataset) + cfg.train.batch_size - 1) // cfg.train.batch_size)
    steps_per_epoch = max(1, (micro_per_epoch + cfg.train.grad_accum - 1) // cfg.train.grad_accum)
    total_steps = steps_per_epoch * cfg.train.epochs
    warmup_steps = int(total_steps * cfg.train.warmup_ratio)

    print("\n=== Model ===")
    model = print_model(cfg, dummy_forward=False)
    model.to(device)
    model.train()
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=_lr_lambda(warmup_steps=warmup_steps, total_steps=max(1, total_steps)),
    )

    step = 0
    start_epoch = 0
    checkpoint_path = cfg.train.session_dir / CHECKPOINT_NAME
    if cfg.train.resume and checkpoint_path.is_file():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        step = int(state["step"])
        start_epoch = int(state["epoch"])
        scheduler.load_state_dict(state["scheduler"])
        optimizer.load_state_dict(state["optimizer"])
        tqdm.write(f"Resumed step {step} epoch {start_epoch} ← {checkpoint_path}")
    elif cfg.train.init_checkpoint is not None:
        init_path = cfg.train.init_checkpoint
        if init_path.is_dir():
            init_path = init_path / CHECKPOINT_NAME
        if not init_path.is_file():
            raise FileNotFoundError(f"init checkpoint not found: {init_path}")
        state = torch.load(init_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        tqdm.write(f"Initialized weights from {init_path} (new session, step 0)")

    _print_setup(cfg, len(dataset), steps_per_epoch, total_steps, warmup_steps, device)
    cfg.train.session_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = cfg.train.session_dir / "viz"
    amp = use_bf16(cfg)
    accum = cfg.train.grad_accum
    optimizer.zero_grad(set_to_none=True)
    micro = 0
    timing_ema = {"data": 0.0, "fwd": 0.0, "bwd": 0.0}
    step_data_sec = step_fwd_sec = step_bwd_sec = 0.0
    batch_wait = time.perf_counter()
    progress = tqdm(
        total=total_steps,
        initial=step,
        desc="Training",
        unit="step",
        dynamic_ncols=True,
    )

    for epoch in range(start_epoch, cfg.train.epochs):
        dataset.epoch = epoch
        for batch in loader:
            data_sec = time.perf_counter() - batch_wait
            images = batch["images"].to(device, non_blocking=True)
            masks = batch["masks"].to(device, non_blocking=True)
            ch = cfg.mask_channels
            opening_masks = torch.clamp(masks[:, ch.doors] + masks[:, ch.windows], max=1.0)
            fwd_start = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits = model(images).mask_logits
                loss = five_mask_loss(
                    logits,
                    masks,
                    opening_masks,
                    mask_channels=cfg.mask_channels,
                    focal_alpha=cfg.loss.focal_alpha,
                    focal_gamma=cfg.loss.focal_gamma,
                    channel_weights=cfg.loss.channel_weights,
                    extra_opening_penalty=cfg.loss.extra_opening_penalty,
                )
            _sync_device(device)
            fwd_sec = time.perf_counter() - fwd_start
            bwd_start = time.perf_counter()
            (loss / accum).backward()
            _sync_device(device)
            bwd_sec = time.perf_counter() - bwd_start
            step_data_sec += data_sec
            step_fwd_sec += fwd_sec
            step_bwd_sec += bwd_sec
            micro += 1
            if micro % accum != 0:
                batch_wait = time.perf_counter()
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            progress.update(1)
            timing_ema["data"] = _ema(timing_ema["data"], step_data_sec)
            timing_ema["fwd"] = _ema(timing_ema["fwd"], step_fwd_sec)
            timing_ema["bwd"] = _ema(timing_ema["bwd"], step_bwd_sec)
            step_data_sec = step_fwd_sec = step_bwd_sec = 0.0
            progress.set_postfix_str(
                f"{loss.item():.4f}, "
                f"{float(optimizer.param_groups[0]['lr']):.2e}, "
                f"{epoch + 1}/{cfg.train.epochs}, "
                f"data={timing_ema['data']:.2f}s "
                f"fwd={timing_ema['fwd']:.2f}s "
                f"bwd={timing_ema['bwd']:.2f}s"
            )
            batch_wait = time.perf_counter()
            if step % cfg.train.viz_steps == 0:
                _save_viz(model, cfg, device, train_count, viz_dir, step)
            if step % cfg.train.save_steps == 0:
                _save_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    epoch=epoch,
                )
                tqdm.write(f"saved → {checkpoint_path}")

    if micro % accum != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        progress.update(1)

    progress.close()
    _save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=step,
        epoch=cfg.train.epochs,
    )
    tqdm.write(f"Saved → {checkpoint_path}")
    tqdm.write(f"train={len(dataset)} steps={step}")


if __name__ == "__main__":
    run(load_config())
