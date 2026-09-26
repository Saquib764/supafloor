"""Train ModernBERT decision model on RACE (cross-entropy over option logits)."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from config import DecisionConfig, datasets_cache_dir, load_config
from data import (
    DecisionCollator,
    ListDecisionDataset,
    RaceDecisionDataset,
    RaceDecisionStreamDataset,
    load_race_decision_dataset,
    load_race_eval_examples,
    race_split_row_count,
    use_local_race,
)
from device import pick_device
from model.decision_model import ModernBERTDecisionModel
from preview import print_model_summary
from registry import load_dotenv
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from checkpoint import load_checkpoint, save_checkpoint, should_write_encoder

TRAINING_STATE_NAME = "training_state.pt"


def use_bf16(cfg: DecisionConfig) -> bool:
    return bool(cfg.train.bf16 and torch.cuda.is_available())


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        synchronize = getattr(torch.mps, "synchronize", None)
        if synchronize is not None:
            synchronize()


def _lr_lambda(*, warmup_steps: int, total_steps: int):
    def fn(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return 1.0
        progress = float(step - warmup_steps) / float(total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return fn


def _checkpoint_dir(cfg: DecisionConfig) -> Path:
    return cfg.train.session_dir / "last"


def _save_training_state(
    path: Path,
    *,
    optimizer: AdamW,
    scheduler: LambdaLR,
    step: int,
    epoch: int,
) -> None:
    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        path,
    )


def _accuracy(logits: torch.Tensor, labels: torch.Tensor, option_mask: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    valid = option_mask.any(dim=-1)
    if not valid.any():
        return 0.0
    correct = (preds[valid] == labels[valid]).sum().item()
    return float(correct) / float(valid.sum().item())


@torch.no_grad()
def evaluate(
    model: ModernBERTDecisionModel,
    loader: DataLoader,
    device: torch.device,
    *,
    amp: bool,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    for batch in loader:
        formatted = batch.formatted
        labels = batch.labels.to(device)
        input_ids = formatted.input_ids.to(device)
        attention_mask = formatted.attention_mask.to(device)
        option_positions = formatted.option_positions.to(device)
        option_mask = formatted.option_mask.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out = model(input_ids, attention_mask, option_positions, option_mask)
            loss = F.cross_entropy(out.logits, labels)
        n = int(labels.size(0))
        total_loss += float(loss.item()) * n
        preds = out.logits.argmax(dim=-1)
        valid = option_mask.any(dim=-1)
        total_correct += int((preds[valid] == labels[valid]).sum().item())
        total_n += int(valid.sum().item())
    if total_n == 0:
        return 0.0, 0.0
    return total_loss / total_n, float(total_correct) / float(total_n)


def _run_eval(
    model: ModernBERTDecisionModel,
    loader: DataLoader,
    device: torch.device,
    *,
    amp: bool,
    sample_count: int,
    step: int | None = None,
) -> tuple[float, float]:
    loss, acc = evaluate(model, loader, device, amp=amp)
    prefix = f"step {step} " if step is not None else ""
    tqdm.write(f"{prefix}eval n={sample_count} loss={loss:.4f} acc={acc:.4f}")
    model.train()
    return loss, acc


def _print_setup(
    cfg: DecisionConfig,
    train_len: int,
    val_len: int,
    steps_per_epoch: int,
    total_steps: int,
    warmup_steps: int,
    device: torch.device,
) -> None:
    print("\n=== Device ===")
    print(f"device:         {device}")
    print(f"compute dtype:  {'bfloat16' if use_bf16(cfg) else 'float32'}")

    print("\n=== Dataset ===")
    print(f"dataset:        {cfg.data.dataset}")
    print(f"source:         {cfg.data.source}")
    print(f"race_subset:    {cfg.data.race_subset}")
    print(f"train_split:    {cfg.data.train_split}")
    print(f"val_split:      {cfg.data.val_split}")
    print(f"cache_dir:      {datasets_cache_dir(cfg)}")
    print(f"train rows:     {train_len:,}")
    print(f"val rows:       {val_len:,}")

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
    print(f"train_encoder:  {cfg.model.train_pretrained_encoder}")
    print(f"resume:         {cfg.train.resume}")
    print(f"save_steps:     {cfg.train.save_steps}")
    print(f"eval_samples:   {cfg.train.eval_samples}")
    print(f"eval_steps:     {cfg.train.eval_steps}")
    print(f"save_full_encoder: {cfg.train.save_full_encoder}")
    print(f"checkpoint mode: {'full encoder' if should_write_encoder(cfg, at_end=False) else 'light (head + markers)'}")
    print()


def run(cfg: DecisionConfig) -> None:
    load_dotenv(cfg.train.session_dir.parent, Path.cwd())
    device = pick_device()
    if use_local_race(cfg):
        train_ds: RaceDecisionDataset | RaceDecisionStreamDataset = RaceDecisionStreamDataset(
            cfg,
            split=cfg.data.train_split,
            shuffle=True,
            seed=cfg.data.seed,
            augment=True,
        )
        train_len = race_split_row_count(cfg, cfg.data.train_split)
        val_len = race_split_row_count(cfg, cfg.data.val_split)
    else:
        train_hf = load_race_decision_dataset(cfg, split=cfg.data.train_split)
        val_hf = load_race_decision_dataset(cfg, split=cfg.data.val_split)
        train_ds = RaceDecisionDataset(train_hf, cfg, augment=True)
        train_len = len(train_ds)
        val_len = len(val_hf)

    eval_examples = load_race_eval_examples(cfg, count=cfg.train.eval_samples)
    if len(eval_examples) < cfg.train.eval_samples:
        tqdm.write(
            f"warning: eval set has {len(eval_examples)} examples "
            f"(requested {cfg.train.eval_samples})"
        )

    print("\n=== Model ===")
    model = print_model_summary(cfg, dummy_forward=False, show_samples=False)
    model.to(device)

    collator = DecisionCollator(model)
    train_shuffle = not use_local_race(cfg)
    loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=train_shuffle,
        num_workers=cfg.data.num_workers,
        collate_fn=collator,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    eval_loader = DataLoader(
        ListDecisionDataset(eval_examples),
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        collate_fn=collator,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    eval_n = len(eval_examples)

    steps_per_epoch = max(1, (train_len + cfg.train.batch_size - 1) // cfg.train.batch_size)
    steps_per_epoch = max(1, (steps_per_epoch + cfg.train.grad_accum - 1) // cfg.train.grad_accum)
    total_steps = steps_per_epoch * cfg.train.epochs
    warmup_steps = int(total_steps * cfg.train.warmup_ratio)

    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=_lr_lambda(warmup_steps=warmup_steps, total_steps=max(1, total_steps)),
    )

    ckpt_dir = _checkpoint_dir(cfg)
    state_path = ckpt_dir / TRAINING_STATE_NAME
    step = 0
    start_epoch = 0

    if cfg.train.resume and ckpt_dir.is_dir() and state_path.is_file():
        model = load_checkpoint(ckpt_dir, cfg)
        model.to(device)
        state = torch.load(state_path, map_location=device, weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step = int(state["step"])
        start_epoch = int(state["epoch"])
        tqdm.write(f"Resumed step {step} epoch {start_epoch} ← {ckpt_dir}")
    elif cfg.train.init_checkpoint is not None:
        init_path = cfg.train.init_checkpoint
        if init_path.is_dir():
            model = load_checkpoint(init_path, cfg)
            model.to(device)
            tqdm.write(f"Initialized weights from {init_path}")

    _print_setup(cfg, train_len, val_len, steps_per_epoch, total_steps, warmup_steps, device)
    if use_local_race(cfg):
        from data import race_local_root

        print(f"local_root:     {race_local_root(cfg)} (streaming JSONL)", flush=True)
    cfg.train.session_dir.mkdir(parents=True, exist_ok=True)
    model.train()
    amp = use_bf16(cfg)
    accum = cfg.train.grad_accum
    optimizer.zero_grad(set_to_none=True)
    micro = 0
    progress = tqdm(
        total=total_steps,
        initial=step,
        desc="Training",
        unit="step",
        dynamic_ncols=True,
    )

    for epoch in range(start_epoch, cfg.train.epochs):
        if isinstance(train_ds, RaceDecisionStreamDataset):
            train_ds.epoch = epoch
        for batch in loader:
            formatted = batch.formatted
            labels = batch.labels.to(device, non_blocking=True)
            input_ids = formatted.input_ids.to(device, non_blocking=True)
            attention_mask = formatted.attention_mask.to(device, non_blocking=True)
            option_positions = formatted.option_positions.to(device, non_blocking=True)
            option_mask = formatted.option_mask.to(device, non_blocking=True)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                out = model(input_ids, attention_mask, option_positions, option_mask)
                loss = F.cross_entropy(out.logits, labels)
            batch_acc = _accuracy(out.logits, labels, option_mask)

            (loss / accum).backward()
            micro += 1
            if micro % accum != 0:
                continue

            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad),
                cfg.train.max_grad_norm,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            progress.update(1)
            progress.set_postfix_str(
                f"loss={loss.item():.4f}, acc={batch_acc:.4f}, "
                f"lr={float(optimizer.param_groups[0]['lr']):.2e}, "
                f"epoch={epoch + 1}/{cfg.train.epochs}"
            )

            if step % cfg.train.eval_steps == 0:
                _run_eval(
                    model,
                    eval_loader,
                    device,
                    amp=amp,
                    sample_count=eval_n,
                    step=step,
                )

            if step % cfg.train.save_steps == 0:
                try:
                    save_checkpoint(model, ckpt_dir, at_end=False)
                    _save_training_state(
                        state_path,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        step=step,
                        epoch=epoch,
                    )
                    tqdm.write(f"saved → {ckpt_dir}")
                except OSError as exc:
                    tqdm.write(f"checkpoint save failed (disk full?): {exc}")

        _run_eval(
            model,
            eval_loader,
            device,
            amp=amp,
            sample_count=eval_n,
        )
        tqdm.write(f"epoch {epoch + 1} complete")

    if micro % accum != 0:
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad),
            cfg.train.max_grad_norm,
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        progress.update(1)

    progress.close()
    save_checkpoint(model, ckpt_dir, at_end=True)
    _save_training_state(
        state_path,
        optimizer=optimizer,
        scheduler=scheduler,
        step=step,
        epoch=cfg.train.epochs,
    )
    _run_eval(model, eval_loader, device, amp=amp, sample_count=eval_n)
    tqdm.write(f"Saved → {ckpt_dir}")


if __name__ == "__main__":
    run(load_config())
