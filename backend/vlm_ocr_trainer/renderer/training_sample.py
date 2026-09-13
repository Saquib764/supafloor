"""Shared training sample generation: full max_seq_len budget, label = rendered text."""

from __future__ import annotations

import random

from config import VariationSettings
from renderer.corpus import WordCorpus, sample_document_for_budget
from renderer.paragraph_render import render_paragraph_document
from renderer.text import RenderedSample
from vocab import CharVocab


def _label_fits(text: str, *, vocab: CharVocab, max_seq_len: int, max_text_chars: int) -> bool:
    if not text or len(text) > max_text_chars:
        return False
    return len(vocab.encode(text, add_bos_eos=True)) <= max_seq_len


def generate_training_sample(
    width: int,
    height: int,
    rng: random.Random,
    corpus: WordCorpus,
    variations: VariationSettings,
    vocab: CharVocab,
    *,
    max_seq_len: int,
    paragraph_counts: tuple[int, ...],
) -> RenderedSample:
    """Render text filling the max_seq_len budget; label is only drawn content."""
    max_text_chars = max(1, max_seq_len - 2)
    counts = (1,) if max_text_chars < 200 else paragraph_counts
    max_attempts = 16

    for attempt in range(max_attempts):
        budget = max(1, max_text_chars - attempt // 4)
        paragraphs = sample_document_for_budget(
            rng,
            corpus,
            settings=variations,
            max_chars=budget,
            paragraph_counts=counts,
        )
        if not paragraphs:
            continue
        sample = render_paragraph_document(width, height, rng, paragraphs, variations)
        if _label_fits(
            sample.text,
            vocab=vocab,
            max_seq_len=max_seq_len,
            max_text_chars=max_text_chars,
        ):
            return sample

    for word in sorted(corpus.all_words, key=len):
        for candidate in (word.capitalize() + ".", word[:max_text_chars]):
            if not candidate or len(candidate) > max_text_chars:
                continue
            sample = render_paragraph_document(width, height, rng, [candidate], variations)
            if _label_fits(
                sample.text,
                vocab=vocab,
                max_seq_len=max_seq_len,
                max_text_chars=max_text_chars,
            ):
                return sample

    raise RuntimeError(
        f"failed to render a sample within max_seq_len={max_seq_len} "
        f"({max_text_chars} text chars) after {max_attempts} attempts",
    )
