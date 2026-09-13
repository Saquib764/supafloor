"""Train OCR VLM on synthetic text images."""

from __future__ import annotations

import math
import random
from multiprocessing import Value
from pathlib import Path

import torch
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from benchmark import (
    append_metrics_log,
    build_holdout_loader,
    require_holdout,
    run_benchmark,
    write_benchmark_score,
)
from holdout import holdout_manifest_path
from references import print_benchmark_report
from config import Config, load_config, print_data_schedule
from data import OcrCollator, SyntheticOcrDataset, image_to_tensor
from renderer.canvas import training_viz_row
from renderer.corpus import require_corpus
from renderer.training_sample import generate_training_sample
from device import training_device
from model import OcrVlm, count_parameters, ocr_loss
from preview import print_model
from vocab import CharVocab

CHECKPOINT_NAME = "last.pt"
BEST_CHECKPOINT_NAME = "best.pt"


def use_bf16(cfg: Config) -> bool:
    return bool(cfg.train.bf16 and torch.cuda.is_available())


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
    model: OcrVlm,
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
    model: OcrVlm,
    vocab: CharVocab,
    cfg: Config,
    corpus,
    device: torch.device,
    *,
    seed: int,
    step: int,
) -> Image.Image:
    rng = random.Random(seed)
    settings = cfg.data.at_step(step)
    max_seq_len = cfg.max_seq_len.at(step)
    width, height = rng.choice(settings.image_sizes)
    sample = generate_training_sample(
        width,
        height,
        rng,
        corpus,
        settings.variations,
        vocab,
        max_seq_len=max_seq_len,
        paragraph_counts=cfg.data.paragraph_counts,
    )
    tensor = image_to_tensor(sample.image).unsqueeze(0).to(device)
    generated = model.generate(tensor, max_length=max_seq_len)
    pred = vocab.decode(generated[0].tolist())
    return training_viz_row(tensor[0], sample.text, pred)


def _save_viz(
    model: OcrVlm,
    vocab: CharVocab,
    cfg: Config,
    corpus,
    device: torch.device,
    out_dir: Path,
    step: int,
    *,
    seed: int,
) -> None:
    model.eval()
    step_dir = out_dir / f"step-{step:06d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    for index in range(cfg.train.viz_samples):
        panel = _viz_row(
            model, vocab, cfg, corpus, device, seed=seed + step + index, step=step
        )
        panel.save(step_dir / f"sample-{index:03d}.jpg", format="JPEG", quality=95)
    model.train()
    tqdm.write(f"viz → {step_dir} ({cfg.train.viz_samples} sample(s))")


def _run_step_benchmark(
    model: OcrVlm,
    vocab: CharVocab,
    cfg: Config,
    device: torch.device,
    loader,
    *,
    epoch: int,
    step: int,
    metrics_path: Path,
    best_cer: float,
    best_path: Path,
    checkpoint_path: Path,
    optimizer: AdamW,
    scheduler: LambdaLR,
) -> float:
    result = run_benchmark(
        model,
        loader,
        vocab,
        cfg,
        device,
        desc=f"Benchmark step {step}",
    )
    print_benchmark_report(
        checkpoint=checkpoint_path,
        holdout=holdout_manifest_path(cfg),
        result=result,
        model_name="OcrVlm",
        model_params=count_parameters(model),
        epoch=epoch + 1,
        write=tqdm.write,
    )
    append_metrics_log(
        metrics_path,
        {
            "epoch": epoch + 1,
            "step": step,
            "cer": result.cer,
            "wer": result.wer,
            "seq_acc": result.seq_acc,
            "n": result.n,
        },
    )
    history_path, latest_path = write_benchmark_score(
        cfg,
        result,
        model_params=count_parameters(model),
        checkpoint=checkpoint_path,
        checkpoint_kind="training",
        epoch=epoch + 1,
        step=step,
    )
    tqdm.write(f"score history → {history_path}")
    tqdm.write(f"latest score  → {latest_path}")
    if cfg.train.save_best and result.cer < best_cer:
        _save_checkpoint(
            best_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            epoch=epoch,
        )
        tqdm.write(f"best → {best_path}")
        return result.cer
    return best_cer


def run(cfg: Config) -> None:
    device = training_device()
    vocab = CharVocab()
    corpus = require_corpus(cfg.words.path)
    training_step_value = Value("i", 0)
    epoch_value = Value("i", 0)
    use_workers = cfg.train.num_workers > 0
    prefetch_samples = 0 if use_workers else cfg.train.prefetch_samples
    if prefetch_samples == 0 and not use_workers:
        prefetch_samples = cfg.train.batch_size * 2
    dataset = SyntheticOcrDataset(
        data=cfg.data,
        corpus=corpus,
        batch_size=cfg.train.batch_size,
        max_seq_len=cfg.max_seq_len,
        vocab=vocab,
        samples_per_epoch=cfg.data.samples_per_epoch,
        seed=cfg.data.seed,
        training_step_value=training_step_value,
        epoch_value=epoch_value,
        prefetch_samples=prefetch_samples,
    )
    loader_kwargs: dict[str, object] = {
        "batch_size": cfg.train.batch_size,
        "shuffle": False,
        "num_workers": cfg.train.num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": OcrCollator(vocab.pad_id),
    }
    if use_workers:
        loader_kwargs["prefetch_factor"] = cfg.train.prefetch_factor
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(dataset, **loader_kwargs)
    micro_per_epoch = max(1, (len(dataset) + cfg.train.batch_size - 1) // cfg.train.batch_size)
    steps_per_epoch = max(1, (micro_per_epoch + cfg.train.grad_accum - 1) // cfg.train.grad_accum)
    total_steps = steps_per_epoch * cfg.train.epochs
    warmup_steps = int(total_steps * cfg.train.warmup_ratio)

    print("\n=== Model ===")
    model = print_model(cfg, vocab, dummy_forward=False)
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
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step = int(state["step"])
        start_epoch = int(state["epoch"])
        training_step_value.value = step
        dataset.epoch = start_epoch
        tqdm.write(f"Resumed step {step} epoch {start_epoch} ← {checkpoint_path}")

    print("\n=== Device ===")
    print(f"device:         {device}")
    print(f"compute dtype:  {'bfloat16' if use_bf16(cfg) else 'float32'}")
    print("\n=== Dataset ===")
    print(f"synthetic:      on-the-fly random text lines")
    print_data_schedule(cfg)
    print(f"samples/epoch:  {cfg.data.samples_per_epoch:,}")
    print(f"vocab size:     {vocab.size}")
    benchmark_enabled = cfg.train.benchmark_steps > 0
    holdout_loader = None
    if benchmark_enabled:
        require_holdout(cfg)
        benchmark_batch_size = cfg.train.benchmark_batch_size or cfg.train.batch_size
        holdout_loader = build_holdout_loader(cfg, benchmark_batch_size)
    print("\n=== Holdout ===")
    print(f"dir:            {cfg.holdout.dir}")
    print(f"samples:        {cfg.holdout.num_samples:,}")
    print(f"benchmark:      every {cfg.train.benchmark_steps} step(s)")
    print(f"save_best:      {cfg.train.save_best}")
    print("\n=== Training ===")
    print(f"output_dir:     {cfg.train.session_dir}")
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
    print(f"benchmark_steps:{cfg.train.benchmark_steps}")
    print(f"viz_steps:      {cfg.train.viz_steps}")
    print(f"viz_samples:    {cfg.train.viz_samples}")
    print(f"num_workers:    {cfg.train.num_workers}")
    print(f"prefetch:       factor={cfg.train.prefetch_factor}, samples={prefetch_samples}")
    print(f"parameters:     {count_parameters(model):,}")
    print()

    cfg.train.session_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = cfg.train.session_dir / "viz"
    metrics_path = cfg.train.session_dir / "metrics.jsonl"
    best_path = cfg.train.session_dir / BEST_CHECKPOINT_NAME
    best_cer = float("inf")
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
        dataset.epoch = epoch
        training_step_value.value = step
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            token_ids = batch["token_ids"].to(device, non_blocking=True)
            targets = token_ids[:, 1:]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits = model(images, token_ids)
                loss = ocr_loss(logits, targets, pad_id=vocab.pad_id)
            (loss / accum).backward()
            micro += 1
            if micro % accum != 0:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            training_step_value.value = step
            progress.update(1)
            progress.set_postfix_str(
                f"{loss.item():.4f}, "
                f"{float(optimizer.param_groups[0]['lr']):.2e}, "
                f"{epoch + 1}/{cfg.train.epochs}"
            )
            if step % cfg.train.viz_steps == 0:
                _save_viz(
                    model,
                    vocab,
                    cfg,
                    corpus,
                    device,
                    viz_dir,
                    step,
                    seed=cfg.data.seed or 0,
                )
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
            if (
                benchmark_enabled
                and holdout_loader is not None
                and step % cfg.train.benchmark_steps == 0
            ):
                best_cer = _run_step_benchmark(
                    model,
                    vocab,
                    cfg,
                    device,
                    holdout_loader,
                    epoch=epoch,
                    step=step,
                    metrics_path=metrics_path,
                    best_cer=best_cer,
                    best_path=best_path,
                    checkpoint_path=checkpoint_path,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )
                model.train()

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
    tqdm.write(f"steps={step}")


if __name__ == "__main__":
    run(load_config())
