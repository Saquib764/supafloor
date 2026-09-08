"""Fine-tune SAM 2.1 on floor-plan room point prompts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import Trainer, TrainerCallback, TrainingArguments
from transformers.trainer_callback import PrinterCallback, ProgressCallback

from canvas import apply_masks
from config import Config, datasets_cache_dir, load_config, use_fp16
from data import (
    FloorplanSamDataset,
    SamCollator,
    VizSample,
    build_viz_samples,
    load_train_stream,
    load_viz_rows,
    sample_viz_point,
)
from model import (
    Sam2TrainingWrapper,
    compute_segmentation_loss,
    load_model_and_processor,
    lora_param_count,
    post_process_pred_masks,
    prepare_pixel_values,
    trainable_param_count,
)


class SamProgressCallback(ProgressCallback):
    def __init__(self, trainer: SamTrainer) -> None:
        super().__init__()
        self._trainer = trainer

    def _postfix(self, *, loss: float | None = None, lr: float | None = None, step: int | None = None) -> dict[str, str]:
        postfix: dict[str, str] = {}
        if loss is not None:
            postfix["loss"] = f"{loss:.4f}"
        if lr is not None:
            postfix["lr"] = f"{lr:.2e}"
        elif self._trainer.optimizer is not None:
            postfix["lr"] = f"{self._trainer.optimizer.param_groups[0]['lr']:.2e}"
        if step is not None:
            postfix["step"] = str(step)
        return postfix

    def on_step_end(self, args: TrainingArguments, state: Any, control: Any, **kwargs: Any) -> None:
        super().on_step_end(args, state, control, **kwargs)
        if state.is_world_process_zero and self.training_bar is not None:
            loss = self._trainer._last_batch_loss
            if loss is not None:
                self.training_bar.set_postfix(
                    self._postfix(loss=loss, step=int(state.global_step)),
                )

    def on_log(self, args: TrainingArguments, state: Any, control: Any, logs: dict | None = None, **kwargs: Any) -> None:
        if state.is_world_process_zero and self.training_bar is not None and logs:
            loss = float(logs["loss"]) if "loss" in logs else None
            lr = float(logs["learning_rate"]) if "learning_rate" in logs else None
            postfix = self._postfix(loss=loss, lr=lr)
            if postfix:
                self.training_bar.set_postfix(postfix)


class SamTrainer(Trainer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._last_batch_loss: float | None = None
        self.callback_handler.callbacks = [
            callback
            for callback in self.callback_handler.callbacks
            if not isinstance(callback, (ProgressCallback, PrinterCallback))
        ]
        self.add_callback(SamProgressCallback(self))

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        batch_loss = loss.item()
        if self.args.gradient_accumulation_steps > 1:
            batch_loss *= self.args.gradient_accumulation_steps
        self._last_batch_loss = batch_loss
        return loss


class VizCallback(TrainerCallback):
    def __init__(
        self,
        wrapper: Sam2TrainingWrapper,
        processor: object,
        samples: list[VizSample],
        *,
        cfg: Config,
        device: torch.device,
        out_dir: Path,
        every: int,
    ) -> None:
        self._wrapper = wrapper
        self._processor = processor
        self._samples = samples
        self._cfg = cfg
        self._device = device
        self._out_dir = out_dir
        self._every = every

    def on_step_end(self, args: TrainingArguments, state: Any, control: Any, **kwargs: Any) -> None:
        step = int(state.global_step)
        if step <= 0 or step % self._every != 0:
            return
        save_training_viz(
            self._wrapper,
            self._processor,
            self._samples,
            cfg=self._cfg,
            device=self._device,
            out_dir=self._out_dir,
            step=step,
        )


def _predict_mask(
    wrapper: Sam2TrainingWrapper,
    processor: object,
    sample: VizSample,
    point: tuple[float, float],
    *,
    canvas_size: int,
    device: torch.device,
) -> np.ndarray:
    points_tensor = torch.tensor(
        [[[[point[0], point[1]]]]],
        dtype=torch.float32,
        device=device,
    )
    labels = torch.tensor(
        [[[1]]],
        dtype=torch.int64,
        device=device,
    )
    outputs = wrapper(
        pixel_values=sample.pixel_values.to(device),
        input_points=points_tensor,
        input_labels=labels,
    )
    original_sizes = torch.tensor([[canvas_size, canvas_size]])
    masks = post_process_pred_masks(processor, outputs.pred_masks.cpu(), original_sizes)
    return masks[0, 0].numpy().astype(bool)


def save_training_viz(
    wrapper: Sam2TrainingWrapper,
    processor: object,
    samples: list[VizSample],
    *,
    cfg: Config,
    device: torch.device,
    out_dir: Path,
    step: int,
) -> None:
    if not samples:
        return
    was_training = wrapper.training
    wrapper.eval()
    step_dir = out_dir / f"step-{step:06d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        point = sample_viz_point(
            sample.extraction,
            sample.row_index,
            step,
            canvas_size=cfg.canvas_size,
            point_distance_alpha=cfg.data.point_distance_alpha,
            seed=cfg.data.seed,
        )
        pred_mask = _predict_mask(
            wrapper,
            processor,
            sample,
            point,
            canvas_size=cfg.canvas_size,
            device=device,
        )
        pred_overlay = apply_masks(sample.image, [pred_mask], [point])
        pred_overlay.save(step_dir / f"{sample.stem}.jpg", format="JPEG", quality=95)
    if was_training:
        wrapper.train()
    tqdm.write(f"viz → {step_dir} ({len(samples)} test image(s))")


def _training_device() -> str:
    if torch.cuda.is_available():
        return f"cuda ({torch.cuda.get_device_name(0)})"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps (Apple GPU)"
    return "cpu"


def _print_training_setup(
    cfg: Config,
    model: torch.nn.Module,
    train_rows: int,
    test_rows: int,
    train_ds: FloorplanSamDataset,
    args: TrainingArguments,
) -> None:
    effective_batch = cfg.train.batch_size * cfg.train.grad_accum
    steps_per_epoch = math.ceil(len(train_ds) / cfg.train.batch_size / cfg.train.grad_accum)
    total_steps = steps_per_epoch * cfg.train.epochs

    print("\n=== Device ===")
    print(f"device:         {_training_device()}")
    print(f"compute dtype:  {'float16' if use_fp16(cfg) else 'float32'}")
    print(f"model device:   {next(model.parameters()).device}")

    print("\n=== Dataset ===")
    print(f"dataset:          {cfg.data.dataset}")
    print(f"revision:         {cfg.data.revision or '(default)'}")
    print(f"cache_dir:        {datasets_cache_dir(cfg)}")
    print(f"train rows:       {train_rows:,}")
    print(f"test rows:        {test_rows:,}")
    print(f"test_fraction:    {cfg.data.test_fraction}")
    print(f"canvas_size:      {cfg.canvas_size}")
    print(f"room_slots:       {cfg.data.room_slots}")
    print(f"per_room_var:     {cfg.data.per_room_variation}")
    print(f"point_dist_alpha: {cfg.data.point_distance_alpha}")
    print(f"samples/row:      {cfg.data.variations_per_floorplan}")
    print(f"train samples:    {len(train_ds):,}")
    print(f"data seed:        {cfg.data.seed}")

    print("\n=== Finetune ===")
    print(f"neck:             {cfg.finetune.neck}")
    print(f"prompt_encoder:   {cfg.finetune.prompt_encoder}")
    print(f"mask_decoder:     {cfg.finetune.mask_decoder}")
    print("\n=== LoRA ===")
    print(f"rank:             {cfg.lora.rank} (alpha {2 * cfg.lora.rank})")
    print(f"dropout:          {cfg.lora.dropout}")
    print(f"stage3:           {cfg.lora.stage3}")
    print(f"stage4:           {cfg.lora.stage4}")
    print(f"lora params:      {lora_param_count(model):,}")
    print(f"trainable params: {trainable_param_count(model):,}")

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
    print(f"viz_steps:        {cfg.train.viz_steps}")
    print(f"viz_samples:      {cfg.train.viz_samples}")
    print(f"num_workers:      {cfg.data.num_workers}")
    print(f"resume:           {cfg.train.resume}")
    print(f"from_checkpoint:  {cfg.train.from_checkpoint or '(none)'}")
    print(f"logging_steps:    {args.logging_steps}")
    print()


def _allow_checkpoint_unpickle() -> None:
    """HF Trainer checkpoints include numpy RNG state; PyTorch 2.6+ defaults weights_only=True."""
    _load = torch.load

    def load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _load(*args, **kwargs)

    torch.load = load  # type: ignore[method-assign]


def _resolve_from_checkpoint(path: Path | None) -> Path | None:
    if path is None:
        return None
    checkpoint = path.resolve()
    if not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")
    return checkpoint


def run(cfg: Config) -> None:
    from_checkpoint = _resolve_from_checkpoint(cfg.train.from_checkpoint)
    if from_checkpoint is not None and cfg.train.resume:
        raise ValueError("train.resume must be false when train.from_checkpoint is set")
    if cfg.train.resume:
        _allow_checkpoint_unpickle()
    cache_dir = datasets_cache_dir(cfg)
    train_rows, train_count, test_count = load_train_stream(
        cfg.data.dataset,
        test_fraction=cfg.data.test_fraction,
        revision=cfg.data.revision,
        cache_dir=cache_dir,
    )
    viz_rows = load_viz_rows(
        cfg.data.dataset,
        train_count,
        count=cfg.train.viz_samples,
        revision=cfg.data.revision,
        cache_dir=cache_dir,
    )

    predictor = load_model_and_processor(
        cfg,
        for_training=True,
        checkpoint=from_checkpoint,
    )
    device = predictor.device
    model = Sam2TrainingWrapper(predictor.model)

    def _prepare(image: Image.Image) -> torch.Tensor:
        return prepare_pixel_values(predictor.processor, image, device)

    viz_samples = build_viz_samples(
        viz_rows,
        cfg.canvas_size,
        prepare_pixel_values=_prepare,
    )
    viz_dir = cfg.train.session_dir / "viz"

    train_ds = FloorplanSamDataset(
        train_rows,
        cfg.canvas_size,
        num_rows=train_count,
        room_slots=cfg.data.room_slots,
        per_room_variation=cfg.data.per_room_variation,
        point_distance_alpha=cfg.data.point_distance_alpha,
        seed=cfg.data.seed,
        prepare_pixel_values=_prepare,
    )
    collator = SamCollator()

    steps_per_epoch = math.ceil(len(train_ds) / cfg.train.batch_size / cfg.train.grad_accum)
    total_steps = steps_per_epoch * cfg.train.epochs
    warmup_steps = int(total_steps * cfg.train.warmup_ratio)

    cfg.train.session_dir.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(cfg.train.session_dir),
        num_train_epochs=cfg.train.epochs,
        max_steps=total_steps,
        per_device_train_batch_size=cfg.train.batch_size,
        gradient_accumulation_steps=cfg.train.grad_accum,
        learning_rate=cfg.train.learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=cfg.train.weight_decay,
        fp16=use_fp16(cfg),
        save_steps=cfg.train.save_steps,
        save_total_limit=3,
        eval_strategy="no",
        logging_steps=50,
        max_grad_norm=cfg.train.max_grad_norm,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=cfg.data.num_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
    )

    callbacks = [
        VizCallback(
            model,
            predictor.processor,
            viz_samples,
            cfg=cfg,
            device=device,
            out_dir=viz_dir,
            every=cfg.train.viz_steps,
        )
    ]
    trainer = SamTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=collator,
        compute_loss_func=compute_segmentation_loss,
        callbacks=callbacks,
    )
    _print_training_setup(cfg, predictor.model, train_count, test_count, train_ds, args)
    trainer.train(resume_from_checkpoint=True if cfg.train.resume else None)
    trainer.save_model(str(cfg.train.session_dir))
    predictor.model.save_pretrained(str(cfg.train.session_dir))
    predictor.processor.save_pretrained(str(cfg.train.session_dir))
    print(f"Saved checkpoint → {cfg.train.session_dir}")


def main(argv: list[str] | None = None) -> None:
    import argparse

    from config import CONFIG_PATH

    parser = argparse.ArgumentParser(description="Fine-tune SAM 2.1 on floor-plan rooms")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args(argv)
    run(load_config(args.config))


if __name__ == "__main__":
    main()
