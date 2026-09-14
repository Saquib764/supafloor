"""Benchmark OCR model on the holdout dataset."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from config import Config
from data import HoldoutCollator, HoldoutDataset
from device import training_device
from holdout import holdout_manifest_path
from metrics import BenchmarkResult, benchmark_strings
from model import OcrVlm, count_parameters
from references import format_benchmark_line, format_params, print_benchmark_report
from vocab import CharVocab

BENCHMARK_HISTORY_NAME = "benchmark_history.jsonl"
BENCHMARK_LATEST_NAME = "benchmark_latest.txt"
BENCHMARK_LENGTH_SLACK = 32


def require_holdout(cfg: Config) -> Path:
    path = holdout_manifest_path(cfg)
    if not path.is_file():
        raise FileNotFoundError(
            f"holdout manifest missing: {path}\nRun: uv run vlm-ocr-train holdout"
        )
    return path


def benchmark_max_lengths(
    texts: list[str],
    vocab: CharVocab,
    *,
    cap: int,
    slack: int = BENCHMARK_LENGTH_SLACK,
) -> list[int]:
    return [
        min(len(vocab.encode(text, add_bos_eos=True)) + slack, cap)
        for text in texts
    ]


def build_holdout_dataset(cfg: Config) -> HoldoutDataset | Subset:
    require_holdout(cfg)
    dataset = HoldoutDataset(holdout_manifest_path(cfg), cfg.holdout.dir)
    if cfg.benchmark.num_samples is None:
        return dataset
    count = min(cfg.benchmark.num_samples, len(dataset))
    return Subset(dataset, range(count))


def build_holdout_loader(cfg: Config, batch_size: int) -> DataLoader:
    dataset = build_holdout_dataset(cfg)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=training_device().type == "cuda",
        collate_fn=HoldoutCollator(),
    )


@torch.inference_mode()
def run_benchmark(
    model: OcrVlm,
    loader: DataLoader,
    vocab: CharVocab,
    cfg: Config,
    device: torch.device,
    *,
    desc: str = "Benchmark",
) -> BenchmarkResult:
    model.eval()
    amp = bool(cfg.train.bf16 and device.type == "cuda")
    references: list[str] = []
    hypotheses: list[str] = []
    for batch in tqdm(loader, desc=desc, unit="batch", leave=False):
        images = batch["images"].to(device, non_blocking=True)
        max_lengths = benchmark_max_lengths(
            batch["texts"],
            vocab,
            cap=cfg.benchmark.max_seq_len,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            generated = model.generate(images, max_length=max_lengths)
        for index in range(generated.shape[0]):
            references.append(batch["texts"][index])
            hypotheses.append(vocab.decode(generated[index].tolist()))
    return benchmark_strings(references, hypotheses)


def format_benchmark(result: BenchmarkResult) -> str:
    return format_benchmark_line(result)


def append_metrics_log(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def build_benchmark_record(
    cfg: Config,
    result: BenchmarkResult,
    *,
    model_params: int,
    checkpoint: Path,
    checkpoint_kind: str,
    epoch: int | None = None,
    step: int | None = None,
) -> dict[str, object]:
    return {
        "date": datetime.now(UTC).isoformat(),
        "model": "OcrVlm",
        "model_params": model_params,
        "checkpoint": str(checkpoint),
        "checkpoint_kind": checkpoint_kind,
        "epoch": epoch,
        "step": step,
        "epochs_total": cfg.train.epochs,
        "holdout": str(holdout_manifest_path(cfg)),
        "holdout_samples": result.n,
        "holdout_seed": cfg.holdout.seed,
        "cer": result.cer,
        "wer": result.wer,
        "seq_acc": result.seq_acc,
        "exact": result.exact,
        "char_errors": result.char_errors,
        "char_count": result.char_count,
        "word_errors": result.word_errors,
        "word_count": result.word_count,
        "data_seed": cfg.data.seed,
        "max_text_len": cfg.holdout.max_text_len,
        "font_size": "8-16",
    }


def format_benchmark_score_file(record: dict[str, object]) -> str:
    epoch = record.get("epoch")
    step = record.get("step")
    epochs_total = record.get("epochs_total")
    training = "n/a"
    if epoch is not None and epochs_total is not None:
        step_part = f", step {step}" if step is not None else ""
        training = f"epoch {epoch}/{epochs_total}{step_part}"

    lines = [
        f"date:            {record['date']}",
        f"model:           {record['model']} ({format_params(int(record['model_params']))})",
        f"training:        {training}",
        f"checkpoint:      {record['checkpoint']} ({record['checkpoint_kind']})",
        f"holdout:         {record['holdout']} ({record['holdout_samples']} samples, seed {record['holdout_seed']})",
        f"data_seed:       {record['data_seed']}",
        f"max_text_len:    {record['max_text_len']}",
        f"font_size:       {record['font_size']}",
        "",
        f"CER:             {record['cer']:.4f}",
        f"WER:             {record['wer']:.4f}",
        f"seq_acc:         {record['seq_acc']:.4f} ({record['exact']}/{record['holdout_samples']} exact)",
    ]
    return "\n".join(lines) + "\n"


def write_benchmark_score(
    cfg: Config,
    result: BenchmarkResult,
    *,
    model_params: int,
    checkpoint: Path,
    checkpoint_kind: str,
    epoch: int | None = None,
    step: int | None = None,
) -> tuple[Path, Path]:
    """Append JSON history and overwrite human-readable latest score file."""
    record = build_benchmark_record(
        cfg,
        result,
        model_params=model_params,
        checkpoint=checkpoint,
        checkpoint_kind=checkpoint_kind,
        epoch=epoch,
        step=step,
    )
    session_dir = cfg.train.session_dir
    session_dir.mkdir(parents=True, exist_ok=True)
    history_path = session_dir / BENCHMARK_HISTORY_NAME
    latest_path = session_dir / BENCHMARK_LATEST_NAME
    append_metrics_log(history_path, record)
    latest_path.write_text(format_benchmark_score_file(record), encoding="utf-8")
    return history_path, latest_path


def load_model_from_checkpoint(cfg: Config, checkpoint: Path, device: torch.device) -> OcrVlm:
    vocab = CharVocab()
    model = OcrVlm(vocab)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device)
    return model


def load_checkpoint_training_state(checkpoint: Path, device: torch.device) -> tuple[int | None, int | None]:
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    epoch = state.get("epoch")
    step = state.get("step")
    return (int(epoch) if epoch is not None else None, int(step) if step is not None else None)


def checkpoint_kind_from_path(checkpoint: Path) -> str:
    name = checkpoint.name.lower()
    if name == "best.pt":
        return "best"
    if name == "last.pt":
        return "last"
    return checkpoint.stem


def run(cfg: Config, *, checkpoint: Path | None = None) -> BenchmarkResult:
    device = training_device()
    checkpoint_path = checkpoint or cfg.infer.checkpoint
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    model = load_model_from_checkpoint(cfg, checkpoint_path, device)
    model_params = count_parameters(model)
    epoch, step = load_checkpoint_training_state(checkpoint_path, device)
    batch_size = cfg.benchmark.batch_size or cfg.train.batch_size
    loader = build_holdout_loader(cfg, batch_size)
    result = run_benchmark(model, loader, model.vocab, cfg, device)
    print_benchmark_report(
        checkpoint=checkpoint_path,
        holdout=holdout_manifest_path(cfg),
        result=result,
        model_name="OcrVlm",
        model_params=model_params,
        epoch=epoch,
    )
    history_path, latest_path = write_benchmark_score(
        cfg,
        result,
        model_params=model_params,
        checkpoint=checkpoint_path,
        checkpoint_kind=checkpoint_kind_from_path(checkpoint_path),
        epoch=epoch,
        step=step,
    )
    print(f"score history → {history_path}")
    print(f"latest score  → {latest_path}")
    return result
