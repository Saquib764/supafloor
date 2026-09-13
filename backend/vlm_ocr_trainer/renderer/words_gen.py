"""Generate word-collection JSON for holdout."""

from __future__ import annotations

import random
from pathlib import Path

from config import Config
from renderer.corpus import TopicCollection, WordCorpus, _normalize_word

_TOPIC_NAMES = (
    "city travel",
    "home cooking",
    "weekend sports",
    "garden plants",
    "music concerts",
    "pet care",
    "weather seasons",
    "school supplies",
    "ocean fishing",
    "mountain hiking",
    "car repair",
    "movie nights",
    "coffee shops",
    "birthday parties",
    "farm markets",
    "train rides",
    "beach shells",
    "winter clothes",
    "art museums",
    "board games",
)

_WORD_PARTS = (
    "sun", "river", "stone", "cloud", "maple", "silver", "amber", "quiet", "rapid",
    "north", "south", "green", "blue", "crisp", "warm", "light", "dark", "fresh",
    "road", "path", "field", "house", "table", "chair", "paper", "clock", "train",
    "apple", "bread", "water", "music", "paint", "brush", "plant", "flower", "seed",
    "walk", "run", "jump", "read", "write", "build", "carry", "drive", "visit",
    "market", "garden", "kitchen", "window", "bridge", "forest", "meadow", "valley",
    "ocean", "island", "harbor", "village", "castle", "temple", "school", "office",
    "button", "pocket", "jacket", "basket", "mirror", "candle", "pillow", "blanket",
    "orange", "lemon", "grape", "melon", "pepper", "butter", "cheese", "coffee",
    "piano", "guitar", "violin", "drum", "stage", "ticket", "travel", "ticket",
)

_TOPIC_ADJECTIVES = (
    "urban", "rural", "coastal", "mountain", "desert", "forest", "river", "lake",
    "winter", "summer", "spring", "autumn", "morning", "evening", "local", "global",
    "modern", "classic", "quiet", "busy", "bright", "soft", "fresh", "wild",
    "indoor", "outdoor", "family", "weekend", "daily", "seasonal", "simple", "fancy",
)

_TOPIC_NOUNS = (
    "travel", "cooking", "sports", "garden", "music", "pets", "weather", "school",
    "fishing", "hiking", "repair", "movies", "coffee", "parties", "markets", "trains",
    "beach", "clothes", "museums", "games", "health", "finance", "crafts", "photography",
    "camping", "cycling", "running", "swimming", "reading", "writing", "art", "dance",
)

_SYLLABLES = (
    "ab", "ac", "ad", "af", "ag", "al", "am", "an", "ap", "ar", "as", "at", "av",
    "ba", "be", "bi", "bo", "bu", "ca", "ce", "ci", "co", "cu", "da", "de", "di",
    "do", "du", "fa", "fe", "fi", "fo", "ga", "ge", "gi", "go", "ha", "he", "hi",
    "ho", "ja", "je", "ji", "jo", "ka", "ke", "ki", "ko", "la", "le", "li", "lo",
    "ma", "me", "mi", "mo", "na", "ne", "ni", "no", "pa", "pe", "pi", "po", "ra",
    "re", "ri", "ro", "sa", "se", "si", "so", "ta", "te", "ti", "to", "va", "ve",
    "vi", "vo", "wa", "we", "wi", "wo", "ya", "ye", "yi", "yo", "za", "ze", "zi",
)


def words_path(cfg: Config) -> Path:
    return cfg.words.path


def _topic_names(rng: random.Random, *, count: int) -> list[str]:
    names: set[str] = set(_TOPIC_NAMES)
    while len(names) < count:
        names.add(f"{rng.choice(_TOPIC_ADJECTIVES)} {rng.choice(_TOPIC_NOUNS)}")
    ordered = list(names)
    rng.shuffle(ordered)
    return ordered[:count]


def _topic_words(rng: random.Random, *, count: int) -> tuple[str, ...]:
    words: set[str] = set()
    limit = count * 100
    attempts = 0
    while len(words) < count:
        attempts += 1
        if attempts > limit:
            raise RuntimeError(f"failed to generate {count} unique words")
        roll = rng.random()
        if roll < 0.35:
            word = _normalize_word(rng.choice(_WORD_PARTS))
        elif roll < 0.65:
            left, right = rng.sample(_WORD_PARTS, k=2)
            word = _normalize_word(left + right[: min(5, len(right))])
        else:
            syllables = rng.randint(2, 4)
            word = _normalize_word("".join(rng.choice(_SYLLABLES) for _ in range(syllables)))
        if len(word) >= 3:
            words.add(word)
    return tuple(sorted(words))[:count]


def build_corpus(cfg: Config) -> WordCorpus:
    rng = random.Random(cfg.words.seed)
    topics: list[TopicCollection] = []
    for name in _topic_names(rng, count=cfg.words.num_topics):
        topics.append(
            TopicCollection(
                name=name,
                words=_topic_words(rng, count=cfg.words.words_per_topic),
            )
        )
    return WordCorpus(topics=tuple(topics), punctuation=(".", ",", ";", ":", "!", "?", "-", "(", ")"))


def generate_word_collections(cfg: Config, *, force: bool = False) -> Path:
    path = words_path(cfg)
    if path.is_file() and not force:
        corpus = WordCorpus.load(path)
        print(f"words already exist: {path} ({len(corpus.topics)} topics)")
        print("Use --force to regenerate.")
        return path

    corpus = build_corpus(cfg)
    corpus.save(path)
    print(f"words → {path} ({len(corpus.topics)} topics, {len(corpus.all_words)} words)")
    return path
