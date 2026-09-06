"""LoRA fine-tune Florence-2 on Hugging Face image + loc-token labels."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import Trainer, TrainerCallback, TrainingArguments

from config import CONFIG_PATH, Config, datasets_cache_dir
from data import (
    FlorenceCollator,
    FloorplanFlorenceDataset,
    SampleRegionSize,
    _VARIATIONS_PER_ROOM,
    load_rows,
    prepare_image,
)
from infer import infer_image
from model import load_model_and_processor, save_trainable_projector, use_bf16
from plot import draw_rooms
from preview import _print_model_and_lora

_TEST_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _test_dir() -> Path:
    # Prefer project tree (next to config.yaml); fall back to packaged test/ beside train.py.
    for candidate in (CONFIG_PATH.parent / "test", Path(__file__).resolve().parent / "test"):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Test image dir not found (expected florence_trainer/test or packaged site-packages/test)"
    )


def _load_test_images(image_size: int) -> list[tuple[str, Image.Image]]:
    test_dir = _test_dir()
    paths = sorted(p for p in test_dir.iterdir() if p.suffix.lower() in _TEST_EXTENSIONS)
    if not paths:
        raise ValueError(f"No test images in {test_dir}")
    return [(path.stem, prepare_image(Image.open(path), image_size)) for path in paths]


class TestInferCallback(TrainerCallback):
    def __init__(
        self,
        cfg: Config,
        processor: Any,
        test_images: list[tuple[str, Image.Image]],
    ) -> None:
        self._cfg = cfg
        self._processor = processor
        self._test_images = test_images
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

        step_dir = self._test_dir / f"step-{step:06d}"
        for name, image in self._test_images:
            payload = infer_image(model, self._processor, self._cfg, image.copy())
            out_dir = step_dir / name
            out_dir.mkdir(parents=True, exist_ok=True)

            payload["image"].save(out_dir / "input.jpg", format="JPEG", quality=95)
            (out_dir / "prompts.txt").write_text("\n\n".join(payload["prompts"]), encoding="utf-8")
            (out_dir / "output.txt").write_text(payload["text"], encoding="utf-8")
            if self._cfg.train.test_infer_render:
                overlay = draw_rooms(payload["image"], payload["rooms"])
                overlay.save(out_dir / "overlay.jpg", format="JPEG", quality=95)
                (out_dir / "rooms.json").write_text(
                    json.dumps({"rooms": payload["rooms"], "text": payload["text"]}, indent=2),
                    encoding="utf-8",
                )
                print(
                    f"Test infer step {step} {name} → {out_dir} "
                    f"({len(payload['rooms'])} rooms)"
                )
            else:
                print(f"Test infer step {step} {name} → {out_dir}")


class SampleRegionCallback(TrainerCallback):
    def __init__(self, sample_region: SampleRegionSize, refresh_steps: int) -> None:
        self._sample_region = sample_region
        self._refresh_steps = refresh_steps

    def on_step_end(
        self,
        args: TrainingArguments,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> None:
        step = int(state.global_step)
        if step <= 0 or step % self._refresh_steps != 0:
            return
        fraction = self._sample_region.reload()
        print(f"sample_region_size: {fraction} (step {step})")


def _allow_checkpoint_unpickle() -> None:
    """HF Trainer checkpoints include numpy RNG state; PyTorch 2.6+ defaults weights_only=True."""
    _load = torch.load

    def load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _load(*args, **kwargs)

    torch.load = load  # type: ignore[method-assign]


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
    rows: Any,
    train_ds: FloorplanFlorenceDataset,
    test_images: list[tuple[str, Image.Image]],
    args: TrainingArguments,
    sample_region: SampleRegionSize | None = None,
) -> None:
    plan_ids = len(set(rows["plan_id"]))
    floorplans_per_batch = cfg.train.batch_size // cfg.data.variations_per_floorplan
    effective_batch = cfg.train.batch_size * cfg.train.grad_accum
    steps_per_epoch = math.ceil(len(train_ds) / floorplans_per_batch / cfg.train.grad_accum)
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
    print(f"train rows:       {len(rows):,} ({plan_ids:,} plan_ids)")
    print(f"test images:      {', '.join(name for name, _ in test_images)}")
    print(f"image_size:       {cfg.image_size}")
    print(f"canvas_size:      {cfg.canvas_size}")
    print(f"max_label_tokens: {cfg.data.max_label_tokens}")
    print(f"loc_bins:         {cfg.loc_bins}")
    print(f"data seed:        {cfg.data.seed}")
    if sample_region is not None:
        side = sample_region.fraction * cfg.canvas_size
        print(
            f"sample_region:    {sample_region.fraction} "
            f"({side:.1f}px side, refresh every {cfg.train.sample_region_refresh_steps} steps)"
        )

    _print_model_and_lora(model, processor, cfg)

    print("\n=== Training ===")
    print(f"output_dir:       {cfg.train.session_dir}")
    print(f"epochs:           {cfg.train.epochs}")
    print(f"batch_size:       {cfg.train.batch_size} samples ({floorplans_per_batch} floorplans × {cfg.data.variations_per_floorplan})")
    print(f"grad_accum:       {cfg.train.grad_accum}")
    print(f"effective batch:  {effective_batch}")
    print(f"steps/epoch:      {steps_per_epoch:,}")
    print(f"total steps:      {total_steps:,}")
    print(f"learning_rate:    {cfg.train.learning_rate}")
    print(f"warmup_ratio:     {cfg.train.warmup_ratio}")
    print(f"weight_decay:     {cfg.train.weight_decay}")
    print(f"max_grad_norm:    {cfg.train.max_grad_norm}")
    print(f"resume:           {cfg.train.resume}")
    print(f"save_steps:       {cfg.train.save_steps}")
    print(f"test_infer_steps: {cfg.train.test_infer_steps}")
    print(f"test_infer_render: {cfg.train.test_infer_render}")
    print(f"logging_steps:    {args.logging_steps}")
    print()


def run(cfg: Config) -> None:
    if cfg.train.resume:
        _allow_checkpoint_unpickle()
    rows = load_rows(
        cfg.data.dataset,
        revision=cfg.data.revision,
        cache_dir=datasets_cache_dir(cfg),
    )
    model, processor = load_model_and_processor(cfg, for_training=True)
    collator = FlorenceCollator(processor, cfg.data.max_label_tokens)
    sample_region = SampleRegionSize(cfg.train.sample_region_size_path)
    sample_region.reload()
    train_ds = FloorplanFlorenceDataset(
        rows,
        cfg.image_size,
        loc_bins=cfg.loc_bins,
        canvas_size=cfg.canvas_size,
        empty_polygon_prob=cfg.data.empty_polygon_prob,
        excluded_point_prob=cfg.data.excluded_point_prob,
        inside_counts=cfg.data.inside_counts,
        excluded_counts=cfg.data.excluded_counts,
        variations_per_floorplan=cfg.data.variations_per_floorplan,
        seed=cfg.data.seed,
        sample_region=sample_region,
    )
    if cfg.data.variations_per_floorplan % _VARIATIONS_PER_ROOM != 0:
        raise ValueError(
            f"data.variations_per_floorplan ({cfg.data.variations_per_floorplan}) must be divisible by "
            f"{_VARIATIONS_PER_ROOM} (variations per room)"
        )
    if cfg.train.batch_size % cfg.data.variations_per_floorplan != 0:
        raise ValueError(
            f"train.batch_size ({cfg.train.batch_size}) must be divisible by "
            f"{cfg.data.variations_per_floorplan} (variations per floorplan)"
        )
    floorplans_per_batch = cfg.train.batch_size // cfg.data.variations_per_floorplan
    test_images = _load_test_images(cfg.image_size)

    cfg.train.session_dir.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(cfg.train.session_dir),
        num_train_epochs=cfg.train.epochs,
        per_device_train_batch_size=floorplans_per_batch,
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
        TestInferCallback(cfg, processor, test_images),
        SampleRegionCallback(sample_region, cfg.train.sample_region_refresh_steps),
    ]
    trainer = FlorenceTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=collator,
        callbacks=callbacks,
    )
    _print_training_setup(cfg, model, processor, rows, train_ds, test_images, args, sample_region)
    trainer.train(resume_from_checkpoint=True if cfg.train.resume else None)
    trainer.save_model(str(cfg.train.session_dir))
    save_trainable_projector(model, cfg.train.session_dir)
    processor.save_pretrained(str(cfg.train.session_dir))
    print(f"Saved adapter → {cfg.train.session_dir}")
    print(f"train={len(train_ds)} test={len(test_images)}")
