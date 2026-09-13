"""Character-level vocabulary for OCR transcription."""

from __future__ import annotations

import string

PUNCTUATION = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"

# Extra symbols common in research text (beyond ASCII punctuation).
RESEARCH_CHARS = "–—“”’×±≤≥αβμ²°"

SPECIAL_TOKENS = ("<PAD>", "<BOS>", "<EOS>")


def build_charset() -> str:
    chars = (
        string.ascii_lowercase
        + string.ascii_uppercase
        + string.digits
        + PUNCTUATION
        + RESEARCH_CHARS
        + " \n"
    )
    seen: set[str] = set()
    ordered: list[str] = []
    for ch in chars:
        if ch not in seen:
            seen.add(ch)
            ordered.append(ch)
    return "".join(ordered)


CHARSET = build_charset()


class CharVocab:
    def __init__(self) -> None:
        self.itos: list[str] = list(SPECIAL_TOKENS) + list(CHARSET)
        self.stoi: dict[str, int] = {token: index for index, token in enumerate(self.itos)}

    @property
    def pad_id(self) -> int:
        return self.stoi["<PAD>"]

    @property
    def bos_id(self) -> int:
        return self.stoi["<BOS>"]

    @property
    def eos_id(self) -> int:
        return self.stoi["<EOS>"]

    @property
    def size(self) -> int:
        return len(self.itos)

    def encode(self, text: str, *, add_bos_eos: bool = True) -> list[int]:
        ids: list[int] = []
        if add_bos_eos:
            ids.append(self.bos_id)
        for ch in text:
            if ch not in self.stoi:
                raise ValueError(f"character not in vocabulary: {ch!r}")
            ids.append(self.stoi[ch])
        if add_bos_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int] | tuple[int, ...], *, skip_special: bool = True) -> str:
        parts: list[str] = []
        for token_id in ids:
            token = self.itos[int(token_id)]
            if skip_special and token in SPECIAL_TOKENS:
                continue
            parts.append(token)
        return "".join(parts)
