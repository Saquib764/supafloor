"""OCR benchmark metrics: CER, WER, and exact sequence accuracy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


def _levenshtein(a: Sequence[object], b: Sequence[object]) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, left in enumerate(a, start=1):
        curr = [i]
        for j, right in enumerate(b, start=1):
            cost = 0 if left == right else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def _word_error_counts(ref: str, hyp: str) -> tuple[int, int]:
    ref_words = ref.split()
    hyp_words = hyp.split()
    if not ref_words:
        return (0 if not hyp_words else len(hyp_words), 0)
    return _levenshtein(ref_words, hyp_words), len(ref_words)


def _char_error_counts(ref: str, hyp: str) -> tuple[int, int]:
    if not ref:
        return (0 if not hyp else len(hyp), 0)
    return _levenshtein(ref, hyp), len(ref)


@dataclass(frozen=True)
class BenchmarkResult:
    cer: float
    wer: float
    seq_acc: float
    n: int
    char_errors: int
    char_count: int
    word_errors: int
    word_count: int
    exact: int


def benchmark_strings(references: list[str], hypotheses: list[str]) -> BenchmarkResult:
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")
    char_errors = char_count = word_errors = word_count = exact = 0
    for ref, hyp in zip(references, hypotheses, strict=True):
        c_err, c_len = _char_error_counts(ref, hyp)
        w_err, w_len = _word_error_counts(ref, hyp)
        char_errors += c_err
        char_count += c_len
        word_errors += w_err
        word_count += w_len
        exact += int(ref == hyp)
    n = len(references)
    cer = char_errors / char_count if char_count else 0.0
    wer = word_errors / word_count if word_count else 0.0
    seq_acc = exact / n if n else 0.0
    return BenchmarkResult(
        cer=cer,
        wer=wer,
        seq_acc=seq_acc,
        n=n,
        char_errors=char_errors,
        char_count=char_count,
        word_errors=word_errors,
        word_count=word_count,
        exact=exact,
    )
