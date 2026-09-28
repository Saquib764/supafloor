"""Train ModernBERT decision model on RACE (BCE on expanded candidate rows)."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from config import DecisionConfig, datasets_cache_dir, load_config
from data import (
    EvalCollator,
    ExpandedMapTrainDataset,
    ExpandedTrainStreamDataset,
    ListDecisionDataset,
    TrainCollator,
    RaceDecisionDataset,
    load_race_decision_dataset,
    ensure_eval_examples,
    ensure_test_train_examples,
    race_split_row_count,
    use_local_race,
)
from device import pick_device
from model.decision_model import ModernBERTDecisionModel, build_model
from model.freeze import encoder_finetune_param_groups
from model.segment_markers import marker_embedding_param_groups
from model.params import parameter_breakdown
from registry import load_dotenv
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from checkpoint import load_checkpoint, save_checkpoint

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


def _optimizer_param_groups(
    model: ModernBERTDecisionModel,
    cfg: DecisionConfig,
) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = [
        {
            "params": list(model.score_head.parameters()),
            "lr": cfg.train.head_learning_rate,
        },
    ]
    groups.extend(encoder_finetune_param_groups(model))
    groups.extend(marker_embedding_param_groups(model))
    if model.train_cls_embedding:
        embed = model.encoder.get_input_embeddings()
        if embed is not None and embed.weight.requires_grad:
            groups.append({"params": [embed.weight], "lr": cfg.train.cls_embedding_learning_rate})
    return groups


def _sum_random_chance(option_mask: torch.Tensor) -> float:
    counts = option_mask.sum(dim=-1)
    valid = counts > 0
    if not valid.any():
        return 0.0
    return float((1.0 / counts[valid].float()).sum().item())


def _batch_correct(
    logits: torch.Tensor,
    labels: torch.Tensor,
    option_mask: torch.Tensor,
) -> tuple[int, int]:
    preds = logits.argmax(dim=-1)
    valid = option_mask.any(dim=-1)
    if not valid.any():
        return 0, 0
    correct = int((preds[valid] == labels[valid]).sum().item())
    return correct, int(valid.sum().item())


@torch.no_grad()
def evaluate(
    model: ModernBERTDecisionModel,
    loader: DataLoader,
    device: torch.device,
    *,
    amp: bool,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_random = 0.0
    total_n = 0
    for batch in loader:
        formatted = batch.formatted
        labels = batch.labels.to(device)
        input_ids = formatted.input_ids.to(device)
        attention_mask = formatted.attention_mask.to(device)
        option_mask = formatted.option_mask.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out = model(input_ids, attention_mask, option_mask)
            gold_logits = out.logits.gather(1, labels.unsqueeze(1)).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(
                gold_logits,
                torch.ones_like(gold_logits),
            )
        batch_n = int(labels.size(0))
        total_loss += float(loss.item()) * batch_n
        correct, n = _batch_correct(out.logits, labels, option_mask)
        total_correct += correct
        total_random += _sum_random_chance(option_mask)
        total_n += n
    if total_n == 0:
        return 0.0, 0.0, 0.0
    return (
        total_loss / total_n,
        float(total_correct) / float(total_n),
        total_random / total_n,
    )


def _run_eval(
    model: ModernBERTDecisionModel,
    loader: DataLoader,
    device: torch.device,
    *,
    amp: bool,
    sample_count: int,
    step: int | None = None,
) -> tuple[float, float, float]:
    loss, acc, rand_acc = evaluate(model, loader, device, amp=amp)
    prefix = f"step {step} " if step is not None else ""
    tqdm.write(
        f"{prefix}eval n={sample_count} loss={loss:.4f} acc={acc:.4f} "
        f"rand={rand_acc:.4f}"
    )
    model.train()
    return loss, acc, rand_acc


def _print_setup(
    cfg: DecisionConfig,
    model: ModernBERTDecisionModel,
    train_len: int,
    val_len: int,
    steps_per_epoch: int,
    total_steps: int,
    warmup_steps: int,
    device: torch.device,
) -> None:
    counts = parameter_breakdown(model)
    print("\n=== Parameters ===")
    print(f"total:          {counts.total:,}")
    print(f"trainable:      {counts.trainable:,}")

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
    print(f"head_lr:        {cfg.train.head_learning_rate}")
    print(f"cls_embed_lr:   {cfg.train.cls_embedding_learning_rate}")
    print(f"train_cls_emb:  {cfg.train.train_cls_embedding}")
    print(f"head_dropout:   {cfg.model.score_head_dropout}")
    n_ft = model.finetune_last_n_layers
    if n_ft:
        print(f"encoder_ft:     last {n_ft} block(s) (per-layer train toggles)")
        print(f"encoder_train:  {list(model.encoder_layer_train)}")
        print(f"encoder_lrs:    {list(model.encoder_finetune_lrs)}")
    else:
        print("encoder_ft:     frozen")
    print(f"marker_train:   {list(model.marker_embedding_train)}")
    print(f"marker_lrs:     {list(model.marker_embedding_lrs)}")
    print(f"resume:         {cfg.train.resume}")
    print(f"save_steps:     {cfg.train.save_steps}")
    print(f"eval_samples:   {cfg.train.eval_samples}")
    print(f"eval_steps:     {cfg.train.eval_steps}")
    print(f"eval_seed:      {cfg.train.eval_seed}")
    print(f"test_mode:      {cfg.train.test_mode}")
    if cfg.train.test_mode:
        print(f"test_samples:   {cfg.train.test_samples}")
    print(
        f"variations:     {cfg.train.variations} "
        f"(1 gold + {max(0, cfg.train.variations - 1)} wrong rows per example)",
    )
    print("checkpoint mode: full encoder (always)")
    print()


def run(cfg: DecisionConfig) -> None:
    load_dotenv(cfg.train.session_dir.parent, Path.cwd())
    device = pick_device()
    test_train_fixture: Path | None = None
    if cfg.train.test_mode:
        train_examples, test_train_fixture = ensure_test_train_examples(cfg)
        base_ds = ListDecisionDataset(train_examples)
        train_ds = ExpandedMapTrainDataset(
            base_ds,
            variations=cfg.train.variations,
            seed=cfg.data.seed,
            augment=cfg.data.train_augment,
        )
        train_len = len(train_ds)
        if use_local_race(cfg):
            val_len = race_split_row_count(cfg, cfg.data.val_split)
        else:
            val_len = len(load_race_decision_dataset(cfg, split=cfg.data.val_split))
    elif use_local_race(cfg):
        train_ds = ExpandedTrainStreamDataset(
            cfg,
            split=cfg.data.train_split,
            variations=cfg.train.variations,
            shuffle=True,
            seed=cfg.data.seed,
        )
        train_len = race_split_row_count(cfg, cfg.data.train_split) * cfg.train.variations
        val_len = race_split_row_count(cfg, cfg.data.val_split)
    else:
        train_hf = load_race_decision_dataset(cfg, split=cfg.data.train_split)
        val_hf = load_race_decision_dataset(cfg, split=cfg.data.val_split)
        base_ds = RaceDecisionDataset(train_hf, cfg)
        train_ds = ExpandedMapTrainDataset(
            base_ds,
            variations=cfg.train.variations,
            seed=cfg.data.seed,
            augment=cfg.data.train_augment,
        )
        train_len = len(train_ds)
        val_len = len(val_hf)

    eval_examples, eval_fixture_path = ensure_eval_examples(cfg)
    if len(eval_examples) < cfg.train.eval_samples:
        tqdm.write(
            f"warning: eval set has {len(eval_examples)} examples "
            f"(requested {cfg.train.eval_samples})"
        )

    ckpt_dir = _checkpoint_dir(cfg)
    state_path = ckpt_dir / TRAINING_STATE_NAME
    step = 0
    start_epoch = 0
    training_state: dict[str, object] | None = None

    if cfg.train.resume and ckpt_dir.is_dir() and state_path.is_file():
        model = load_checkpoint(ckpt_dir, cfg)
        training_state = torch.load(state_path, map_location="cpu", weights_only=False)
        step = int(training_state["step"])
        start_epoch = int(training_state["epoch"])
    elif cfg.train.init_checkpoint is not None:
        init_path = cfg.train.init_checkpoint
        if init_path.is_dir():
            model = load_checkpoint(init_path, cfg, trainability_from_config=True)
            tqdm.write(f"Initialized weights from {init_path}")
        else:
            raise FileNotFoundError(f"init_checkpoint not found: {init_path}")
    else:
        model = build_model(cfg)

    model.to(device)

    train_collator = TrainCollator(model)
    eval_collator = EvalCollator(model)
    train_shuffle = cfg.train.test_mode or not use_local_race(cfg)
    loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=train_shuffle,
        num_workers=cfg.data.num_workers,
        collate_fn=train_collator,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    eval_loader = DataLoader(
        ListDecisionDataset(eval_examples),
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        collate_fn=eval_collator,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    eval_n = len(eval_examples)

    steps_per_epoch = max(1, (train_len + cfg.train.batch_size - 1) // cfg.train.batch_size)
    steps_per_epoch = max(1, (steps_per_epoch + cfg.train.grad_accum - 1) // cfg.train.grad_accum)
    total_steps = steps_per_epoch * cfg.train.epochs
    warmup_steps = int(total_steps * cfg.train.warmup_ratio)

    optimizer = AdamW(
        _optimizer_param_groups(model, cfg),
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=_lr_lambda(warmup_steps=warmup_steps, total_steps=max(1, total_steps)),
    )

    if training_state is not None:
        try:
            optimizer.load_state_dict(training_state["optimizer"])  # type: ignore[arg-type]
            scheduler.load_state_dict(training_state["scheduler"])  # type: ignore[arg-type]
        except (ValueError, KeyError) as exc:
            tqdm.write(f"warning: could not restore optimizer ({exc}); using fresh optimizer")
        tqdm.write(f"Resumed step {step} epoch {start_epoch} ← {ckpt_dir}")

    _print_setup(
        cfg, model, train_len, val_len, steps_per_epoch, total_steps, warmup_steps, device
    )
    print(f"eval_fixture:   {eval_fixture_path} ({eval_n} fixed rows)", flush=True)
    if test_train_fixture is not None:
        print(f"test_train:     {test_train_fixture} ({train_len} fixed rows)", flush=True)
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
        if isinstance(train_ds, ExpandedTrainStreamDataset):
            train_ds.epoch = epoch
        for batch in loader:
            formatted = batch.formatted
            labels = batch.labels.to(device, non_blocking=True)
            input_ids = formatted.input_ids.to(device, non_blocking=True)
            attention_mask = formatted.attention_mask.to(device, non_blocking=True)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits = model.score_gold(input_ids, attention_mask)
                loss = F.binary_cross_entropy_with_logits(logits, labels)

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
                f"loss={loss.item():.4f}, "
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
