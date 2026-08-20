"""LoRA fine-tune Florence-2 on Hugging Face image + loc-token labels."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import Trainer, TrainerCallback, TrainingArguments

from config import Config, datasets_cache_dir
from data import FlorenceCollator, FloorplanFlorenceDataset, load_rows, prepare_image, split_by_plan_id
from infer import infer_image
from model import load_model_and_processor, use_bf16
from plot import draw_rooms
from preview import _print_model_and_lora


class TestInferCallback(TrainerCallback):
    def __init__(
        self,
        cfg: Config,
        processor: Any,
        test_image: Image.Image,
    ) -> None:
        self._cfg = cfg
        self._processor = processor
        self._test_image = test_image
        self._test_dir = cfg.train.session_dir / "test"

    def on_step_end(
        self,
        args: TrainingArguments,
        state: Any,
        control: Any,
        model: Any = None,
        **kwargs: Any,
    ) -> None:
        step = int(state.global_step)
        if step <= 0 or step % self._cfg.train.test_infer_steps != 0:
            return
        if model is None:
            return

        payload = infer_image(model, self._processor, self._cfg, self._test_image.copy())
        out_dir = self._test_dir / f"step-{step:06d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        payload["image"].save(out_dir / "input.jpg", format="JPEG", quality=95)
        overlay = draw_rooms(payload["image"], payload["rooms"])
        overlay.save(out_dir / "overlay.jpg", format="JPEG", quality=95)
        (out_dir / "rooms.json").write_text(
            json.dumps({"rooms": payload["rooms"], "text": payload["text"]}, indent=2),
            encoding="utf-8",
        )
        print(f"Test infer step {step} → {out_dir} ({len(payload['rooms'])} rooms)")


class FlorenceTrainer(Trainer):
    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs)
        batch_loss = loss.item()
        if self.args.gradient_accumulation_steps > 1:
            batch_loss *= self.args.gradient_accumulation_steps
        print(f"batch loss: {batch_loss:.4f} (step {self.state.global_step})")
        return loss


def _training_device() -> str:
    if torch.cuda.is_available():
        return f"cuda ({torch.cuda.get_device_name(0)})"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps (Apple GPU)"
    return "cpu"


def _print_training_setup(
    cfg: Config,
    model: Any,
    processor: Any,
    train_rows: Any,
    val_rows: Any,
    train_ds: FloorplanFlorenceDataset,
    eval_ds: FloorplanFlorenceDataset | None,
    args: TrainingArguments,
) -> None:
    train_plans = len(set(train_rows["plan_id"]))
    val_plans = len(set(val_rows["plan_id"]))
    effective_batch = cfg.train.batch_size * cfg.train.grad_accum
    steps_per_epoch = math.ceil(len(train_ds) / effective_batch)
    total_steps = steps_per_epoch * cfg.train.epochs

    print("\n=== Device ===")
    print(f"device:         {_training_device()}")
    print(f"cuda available: {torch.cuda.is_available()}")
    mps = getattr(torch.backends, "mps", None)
    print(f"mps available:  {mps is not None and mps.is_available()}")
    print(f"compute dtype:  {'bfloat16' if use_bf16(cfg) else 'float32'}")
    print(f"model device:   {next(model.parameters()).device}")

    print("\n=== Dataset ===")
    print(f"dataset:          {cfg.data.dataset}")
    print(f"revision:         {cfg.data.revision or '(default)'}")
    print(f"cache_dir:        {datasets_cache_dir(cfg)}")
    print(f"rows:             {len(train_rows) + len(val_rows):,} total")
    print(f"train rows:       {len(train_rows):,} ({train_plans:,} plan_ids)")
    print(f"val rows:         {len(val_rows):,} ({val_plans:,} plan_ids)")
    print(f"train_ratio:      {cfg.data.train_ratio}")
    print(f"split seed:       {cfg.data.seed}")
    print(f"task_prompt:      {cfg.task_prompt!r}")
    print(f"image_size:       {cfg.image_size}")
    print(f"max_label_tokens: {cfg.data.max_label_tokens}")
    print(f"loc_bins:         {cfg.loc_bins}")

    _print_model_and_lora(model, processor, cfg)

    print("\n=== Training ===")
    print(f"output_dir:       {cfg.train.session_dir}")
    print(f"epochs:           {cfg.train.epochs}")
    print(f"batch_size:       {cfg.train.batch_size}")
    print(f"grad_accum:       {cfg.train.grad_accum}")
    print(f"effective batch:  {effective_batch}")
    print(f"steps/epoch:      {steps_per_epoch:,}")
    print(f"total steps:      {total_steps:,}")
    print(f"learning_rate:    {cfg.train.learning_rate}")
    print(f"warmup_ratio:     {cfg.train.warmup_ratio}")
    print(f"weight_decay:     {cfg.train.weight_decay}")
    print(f"max_grad_norm:    {cfg.train.max_grad_norm}")
    print(f"save_steps:       {cfg.train.save_steps}")
    print(f"test_infer_steps: {cfg.train.test_infer_steps}")
    print(f"eval dataset:     {len(eval_ds) if eval_ds is not None else 0} rows")
    print(f"logging_steps:    {args.logging_steps}")
    print()


def run(cfg: Config) -> None:
    rows = load_rows(
        cfg.data.dataset,
        revision=cfg.data.revision,
        cache_dir=datasets_cache_dir(cfg),
    )
    train_rows, val_rows = split_by_plan_id(
        rows,
        train_ratio=cfg.data.train_ratio,
        seed=cfg.data.seed,
    )
    model, processor = load_model_and_processor(cfg, for_training=True)
    collator = FlorenceCollator(processor, cfg.data.max_label_tokens)
    train_ds = FloorplanFlorenceDataset(train_rows, cfg.task_prompt, cfg.image_size)
    eval_ds = (
        FloorplanFlorenceDataset(val_rows, cfg.task_prompt, cfg.image_size)
        if len(val_rows)
        else None
    )

    test_row = val_rows[0] if len(val_rows) else train_rows[0]
    test_image = prepare_image(test_row["image"], cfg.image_size)

    cfg.train.session_dir.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(cfg.train.session_dir),
        num_train_epochs=cfg.train.epochs,
        per_device_train_batch_size=cfg.train.batch_size,
        gradient_accumulation_steps=cfg.train.grad_accum,
        learning_rate=cfg.train.learning_rate,
        warmup_ratio=cfg.train.warmup_ratio,
        weight_decay=cfg.train.weight_decay,
        bf16=use_bf16(cfg),
        save_steps=cfg.train.save_steps,
        save_total_limit=3,
        eval_strategy="no",
        logging_steps=min(50, cfg.train.test_infer_steps),
        max_grad_norm=cfg.train.max_grad_norm,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to="none",
        dataloader_num_workers=0,
    )
    callbacks = [
        TestInferCallback(cfg, processor, test_image),
    ]
    trainer = FlorenceTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        callbacks=callbacks,
    )
    _print_training_setup(cfg, model, processor, train_rows, val_rows, train_ds, eval_ds, args)
    trainer.train()
    trainer.save_model(str(cfg.train.session_dir))
    processor.save_pretrained(str(cfg.train.session_dir))
    print(f"Saved adapter → {cfg.train.session_dir}")
    print(f"train={len(train_ds)} val={len(eval_ds) if eval_ds is not None else 0}")
