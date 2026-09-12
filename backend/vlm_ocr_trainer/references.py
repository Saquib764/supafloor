"""Published OCR reference models for benchmark context."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from metrics import BenchmarkResult


@dataclass(frozen=True)
class ReferenceModel:
    name: str
    params: int | None
    release_date: str | None
    cer: float | None
    wer: float | None
    seq_acc: float | None
    benchmark_note: str


# External references, smallest → largest. Includes Sarvam AI and Datalab Chandra.
# Metrics are published values where available; NA when not reported comparably.
REFERENCE_MODELS: tuple[ReferenceModel, ...] = (
    ReferenceModel(
        name="PaddleOCR PP-OCRv4 mobile",
        params=4_700_000,
        release_date="2023-08",
        cer=None,
        wer=None,
        seq_acc=None,
        benchmark_note="Mobile recognition model; IAM CER/WER not published",
    ),
    ReferenceModel(
        name="CRNN+CTC",
        params=8_300_000,
        release_date="2015-09",
        cer=0.115,
        wer=None,
        seq_acc=None,
        benchmark_note="IAM offline HTR baseline (Shi et al., 2015)",
    ),
    ReferenceModel(
        name="docTR fast",
        params=13_900_000,
        release_date="2021-06",
        cer=None,
        wer=None,
        seq_acc=None,
        benchmark_note="Mindee docTR; IAM handwritten scores not published",
    ),
    ReferenceModel(
        name="EasyOCR",
        params=None,
        release_date="2019-06",
        cer=None,
        wer=None,
        seq_acc=None,
        benchmark_note="CRAFT + CRNN stack; param count not reported",
    ),
    ReferenceModel(
        name="TrOCR-small",
        params=62_092_994,
        release_date="2021-09",
        cer=None,
        wer=0.060,
        seq_acc=None,
        benchmark_note="IAM handwritten WER (Microsoft, 2021)",
    ),
    ReferenceModel(
        name="TrOCR-base",
        params=139_104_258,
        release_date="2021-09",
        cer=None,
        wer=0.049,
        seq_acc=None,
        benchmark_note="IAM handwritten WER (Microsoft, 2021)",
    ),
    ReferenceModel(
        name="TrOCR-large",
        params=333_903_874,
        release_date="2021-09",
        cer=None,
        wer=0.044,
        seq_acc=None,
        benchmark_note="IAM handwritten WER (Microsoft, 2021)",
    ),
    ReferenceModel(
        name="GOT-OCR 2.0",
        params=580_000_000,
        release_date="2024-09",
        cer=None,
        wer=None,
        seq_acc=None,
        benchmark_note="Unified OCR-2.0 model; IAM CER/WER not published",
    ),
    ReferenceModel(
        name="Sarvam-1 (Sarvam AI)",
        params=2_000_000_000,
        release_date="2024-10",
        cer=None,
        wer=None,
        seq_acc=None,
        benchmark_note="Indic text LLM; OCR CER/WER/seq_acc not published",
    ),
    ReferenceModel(
        name="Sarvam 3B sovereign (Sarvam AI)",
        params=3_000_000_000,
        release_date="2026-02",
        cer=None,
        wer=None,
        seq_acc=None,
        benchmark_note="Sovereign base LM; OCR scores via Sarvam Vision fine-tune",
    ),
    ReferenceModel(
        name="Sarvam Vision (Sarvam AI)",
        params=3_000_000_000,
        release_date="2026-02",
        cer=None,
        wer=0.122,
        seq_acc=0.843,
        benchmark_note=(
            "WER: macro avg on Sarvam Indic OCR Bench (22 langs); "
            "seq_acc: olmOCR-Bench English pass rate (84.3%)"
        ),
    ),
    ReferenceModel(
        name="Chandra OCR 2 (Datalab)",
        params=4_000_000_000,
        release_date="2026-03",
        cer=None,
        wer=None,
        seq_acc=0.859,
        benchmark_note="seq_acc: olmOCR-Bench overall pass rate (85.9%)",
    ),
    ReferenceModel(
        name="Chandra OCR 1 (Datalab)",
        params=9_000_000_000,
        release_date="2025-10",
        cer=None,
        wer=None,
        seq_acc=0.831,
        benchmark_note="seq_acc: olmOCR-Bench overall pass rate (83.1%)",
    ),
)


def format_params(params: int | None) -> str:
    if params is None:
        return "NA"
    if params >= 1_000_000:
        return f"{params / 1e6:.2f}M"
    if params >= 1_000:
        return f"{params / 1e3:.1f}K"
    return str(params)


def format_metric(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "NA"


def format_release_date(release_date: str | None) -> str:
    return release_date if release_date is not None else "NA"


def format_benchmark_line(result: BenchmarkResult) -> str:
    return (
        f"cer={result.cer:.4f}  wer={result.wer:.4f}  "
        f"seq_acc={result.seq_acc:.4f}  ({result.exact}/{result.n} exact)"
    )


def _format_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> list[str]:
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    header_line = " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    divider = "-+-".join("-" * widths[index] for index in range(len(headers)))
    body = [
        " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row))
        for row in rows
    ]
    return [header_line, divider, *body]


def format_reference_table() -> tuple[list[str], list[str]]:
    headers = ("#", "Model", "Params", "Released", "CER", "WER", "seq_acc")
    rows: list[tuple[str, ...]] = []
    notes: list[str] = []
    for index, ref in enumerate(REFERENCE_MODELS, start=1):
        rows.append(
            (
                str(index),
                ref.name,
                format_params(ref.params),
                format_release_date(ref.release_date),
                format_metric(ref.cer),
                format_metric(ref.wer),
                format_metric(ref.seq_acc),
            )
        )
        notes.append(f"  {index}. {ref.benchmark_note}")
    return _format_table(headers, rows), notes


def print_benchmark_report(
    *,
    checkpoint: Path,
    holdout: Path,
    result: BenchmarkResult,
    model_name: str,
    model_params: int,
    epoch: int | None = None,
    write: Callable[[str], None] = print,
) -> None:
    count = len(REFERENCE_MODELS)
    write(f"\n=== OCR benchmark references ({count} models) ===")
    write(
        f"{count} external OCR models (smallest → largest), including Sarvam AI and "
        "Datalab Chandra. "
        "Published scores where available; NA = not reported in comparable form. "
        "None were measured on this holdout."
    )
    table, notes = format_reference_table()
    for line in table:
        write(line)
    write("Benchmark notes:")
    for line in notes:
        write(line)

    write("\n=== Metrics ===")
    write(
        "CER (character error rate): character edit distance / reference length; "
        "lower is better — strong OCR models often reach <0.05 on clean text."
    )
    write(
        "WER (word error rate): word edit distance / reference word count; "
        "lower is better — strong OCR models often reach <0.10 on clean text."
    )
    write(
        "seq_acc (sequence accuracy): fraction of predictions that exactly match "
        "the reference string; higher is better — strong models often exceed 0.90 "
        "on single-line printed text."
    )
    write(
        "A better model on this holdout should show lower CER/WER and higher seq_acc "
        "than an under-trained checkpoint; this holdout is harder (multi-line, fonts, "
        "noise) than typical IAM or document OCR benchmarks."
    )

    write("\n=== Results (this holdout) ===")
    if epoch is not None:
        write(f"epoch:       {epoch}")
    write(f"model:       {model_name} ({format_params(model_params)})")
    write(f"checkpoint:  {checkpoint}")
    write(f"holdout:     {holdout}")
    write(f"samples:     {result.n:,}")
    write(format_benchmark_line(result))
