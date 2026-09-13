"""Word collections and paragraph token sampling for holdout rendering."""

from __future__ import annotations

import json
import random
import string
from dataclasses import dataclass
from pathlib import Path

from config import VariationSettings, default_variation_settings

_ATTACH_PUNCT = frozenset(".,;:!?)°²")
_OPEN_PUNCT = frozenset("([“")
_CLOSE_ATTACH = frozenset("”’")


def _is_content_token(token: str) -> bool:
    return bool(token) and token.isalnum()


def _ends_with_content(text: str) -> bool:
    return bool(text) and text[-1].isalnum()


@dataclass(frozen=True)
class TopicCollection:
    name: str
    words: tuple[str, ...]


@dataclass(frozen=True)
class WordCorpus:
    topics: tuple[TopicCollection, ...]
    punctuation: tuple[str, ...]

    @property
    def all_words(self) -> tuple[str, ...]:
        words: list[str] = []
        for topic in self.topics:
            words.extend(topic.words)
        return tuple(words)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "topics": [
                {"name": topic.name, "words": list(topic.words)}
                for topic in self.topics
            ],
            "punctuation": list(self.punctuation),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> WordCorpus:
        raw = json.loads(path.read_text(encoding="utf-8"))
        topics_raw = raw.get("topics") or []
        topics: list[TopicCollection] = []
        for entry in topics_raw:
            name = str(entry["name"]).strip()
            words = tuple(
                _normalize_word(word)
                for word in entry.get("words") or []
                if _normalize_word(word)
            )
            if name and words:
                topics.append(TopicCollection(name=name, words=words))
        if not topics:
            raise ValueError(f"no topics in word corpus: {path}")
        punctuation = tuple(
            ch
            for ch in (raw.get("punctuation") or [".", ",", ";", ":", "!", "?", "-", "(", ")"])
            if ch in string.punctuation and ch in _vocab_punctuation()
        )
        if not punctuation:
            punctuation = (".", ",", ";", ":", "!", "?", "-")
        return cls(topics=tuple(topics), punctuation=punctuation)


def require_corpus(path: Path) -> WordCorpus:
    if not path.is_file():
        raise FileNotFoundError(
            f"word corpus missing: {path}\nRun: uv run vlm-ocr-train words"
        )
    return WordCorpus.load(path)


def _vocab_punctuation() -> str:
    from vocab import PUNCTUATION

    return PUNCTUATION


def _normalize_word(word: str) -> str:
    cleaned = "".join(ch for ch in word.strip().lower() if ch in string.ascii_lowercase)
    return cleaned if len(cleaned) >= 2 else ""


def join_tokens(tokens: list[str]) -> str:
    """Join tokens: space between words/numbers; punctuation attaches without a leading space."""
    if not tokens:
        return ""
    text = tokens[0]
    for token in tokens[1:]:
        if token in _ATTACH_PUNCT or token in _CLOSE_ATTACH:
            text += token
        elif text[-1] in _OPEN_PUNCT:
            text += token
        elif _ends_with_content(text) and _is_content_token(token):
            text += " " + token
        else:
            text += " " + token
    return text


def _sample_research_snippet(rng: random.Random, *, words: tuple[str, ...]) -> str:
    word = rng.choice(words)
    templates = (
        lambda: f"{rng.randint(1, 99)}°C",
        lambda: f"{rng.randint(2, 15)}×",
        lambda: f"±{rng.randint(1, 9)}%",
        lambda: "α–β",
        lambda: f"α = {rng.uniform(0.01, 0.99):.2f}",
        lambda: f"β ≤ {rng.randint(1, 5)}",
        lambda: f"p ≤ 0.{rng.randint(1, 5):02d}",
        lambda: f"≥ {rng.randint(50, 99)}%",
        lambda: f"μ{rng.choice(['g', 'm', 'l', 's'])}",
        lambda: f"{rng.randint(2, 10)}²",
        lambda: f"{rng.randint(1, 9)}—{rng.randint(10, 99)}",
        lambda: f"≤ {rng.randint(1, 20)} mg",
        lambda: f"[{rng.randint(1, 9)}:{rng.randint(10, 99)}]",
        lambda: f"Q{rng.randint(1, 9)}/X{rng.randint(1, 9)}",
        lambda: f"{rng.randint(1, 99)}_{rng.randint(1, 99)}",
        lambda: f"7/{rng.randint(10, 99)}",
        lambda: f"\\{word[:4]}",
        lambda: f"“{word}”",
        lambda: f"{word}’s",
    )
    return rng.choice(templates)()


def _sample_number(rng: random.Random) -> str:
    length = rng.randint(2, 6)
    first = str(rng.randint(1, 9))
    rest = "".join(str(rng.randint(0, 9)) for _ in range(length - 1))
    return first + rest


def _next_token(
    rng: random.Random,
    corpus: WordCorpus,
    *,
    words: tuple[str, ...],
    after_content: bool,
    settings: VariationSettings,
) -> tuple[str, str]:
    roll = rng.random()
    if after_content and roll < settings.research_prob:
        return _sample_research_snippet(rng, words=words), "research"
    if after_content and roll < settings.research_prob + settings.punct_after_content_prob:
        return rng.choice(corpus.punctuation), "punct"
    if not after_content and roll < settings.punct_before_content_prob:
        return rng.choice(corpus.punctuation), "punct"
    if roll < settings.number_prob:
        return _sample_number(rng), "number"
    return rng.choice(words), "word"


def sample_paragraph_tokens(
    rng: random.Random,
    corpus: WordCorpus,
    *,
    min_len: int,
    max_len: int,
    settings: VariationSettings | None = None,
    topic: TopicCollection | None = None,
) -> list[str]:
    settings = settings or default_variation_settings()
    topic = topic or rng.choice(corpus.topics)
    words = topic.words if rng.random() < settings.topic_words_prob else corpus.all_words
    tokens: list[str] = []
    after_content = False
    need_cap = True
    length = 0

    while length < min_len or (length < max_len and rng.random() < settings.continue_prob):
        token, kind = _next_token(
            rng,
            corpus,
            words=words,
            after_content=after_content,
            settings=settings,
        )
        if kind == "word" and need_cap:
            token = token.capitalize()
            need_cap = False
        if kind == "punct" and token in ".!?":
            need_cap = True
            after_content = False
        elif kind in {"word", "number", "research"}:
            after_content = True
        else:
            after_content = False

        if tokens:
            length += 1 if token not in _ATTACH_PUNCT else 0
        length += len(token)
        tokens.append(token)

        if length >= max_len:
            break

    while tokens and tokens[-1] in corpus.punctuation and tokens[-1] not in ".!?":
        tokens.pop()
    if not tokens or tokens[-1] not in ".!?":
        if len(join_tokens([*tokens, "."])) <= max_len:
            tokens.append(".")
    while tokens and len(join_tokens(tokens)) > max_len:
        tokens.pop()
        while tokens and tokens[-1] in corpus.punctuation and tokens[-1] not in ".!?":
            tokens.pop()
    if tokens and tokens[-1] not in ".!?":
        if len(join_tokens([*tokens, "."])) <= max_len:
            tokens.append(".")
    return tokens


def sample_bounded_paragraph(
    rng: random.Random,
    corpus: WordCorpus,
    *,
    max_chars: int,
    settings: VariationSettings | None = None,
) -> str:
    """Sample one paragraph with len(text) <= max_chars."""
    max_chars = max(1, max_chars)
    if max_chars <= 3:
        words = [word for word in corpus.all_words if len(word) <= max_chars]
        return rng.choice(words or corpus.all_words)[:max_chars]

    min_len = 1 if max_chars < 8 else min(max_chars - 1, max(1, max_chars // 2))
    tokens = sample_paragraph_tokens(
        rng,
        corpus,
        min_len=min_len,
        max_len=max_chars,
        settings=settings,
    )
    text = join_tokens(tokens)
    while len(text) > max_chars and tokens:
        tokens.pop()
        while tokens and tokens[-1] in corpus.punctuation and tokens[-1] not in ".!?":
            tokens.pop()
        text = join_tokens(tokens)
    if text and text[-1] not in ".!?" and len(text) + 1 <= max_chars:
        text += "."
    return text[:max_chars]


def sample_document_paragraphs(
    rng: random.Random,
    corpus: WordCorpus,
    *,
    settings: VariationSettings,
    paragraph_counts: tuple[int, ...],
    paragraph_len_min: int,
    paragraph_len_max: int,
) -> list[str]:
    count = rng.choice(paragraph_counts)
    per_min = max(1, paragraph_len_min // count)
    per_max = max(per_min, paragraph_len_max // count)
    paragraphs: list[str] = []
    for _ in range(count):
        target_max = rng.randint(per_min, per_max)
        slack = min(40, max(1, per_max // 2))
        target_min = min(per_min, max(1, target_max - slack))
        tokens = sample_paragraph_tokens(
            rng,
            corpus,
            min_len=target_min,
            max_len=target_max,
            settings=settings,
        )
        paragraphs.append(join_tokens(tokens))
    return paragraphs
