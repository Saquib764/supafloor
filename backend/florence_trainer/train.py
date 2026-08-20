"""LoRA fine-tune Florence-2 on Hugging Face image + loc-token labels."""

from __future__ import annotations

from transformers import Trainer, TrainingArguments

from config import Config, datasets_cache_dir
from data import FlorenceCollator, FloorplanFlorenceDataset, load_rows, split_by_plan_id
from model import load_model_and_processor, use_bf16


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

    cfg.train.output_dir.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(cfg.train.output_dir),
        num_train_epochs=cfg.train.epochs,
        per_device_train_batch_size=cfg.train.batch_size,
        gradient_accumulation_steps=cfg.train.grad_accum,
        learning_rate=cfg.train.learning_rate,
        warmup_ratio=cfg.train.warmup_ratio,
        weight_decay=cfg.train.weight_decay,
        bf16=use_bf16(cfg),
        save_steps=cfg.train.save_steps,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=cfg.train.eval_steps,
        logging_steps=min(50, cfg.train.save_steps),
        max_grad_norm=cfg.train.max_grad_norm,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to="none",
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(str(cfg.train.output_dir))
    processor.save_pretrained(str(cfg.train.output_dir))
    print(f"Saved adapter → {cfg.train.output_dir}")
    print(f"train={len(train_ds)} val={len(eval_ds) if eval_ds is not None else 0}")
