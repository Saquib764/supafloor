"""On-the-fly synthetic OCR samples: random text lines → grayscale image."""

from __future__ import annotations

import random
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import IterableDataset

from vocab import CharVocab

_PRINTED_FONTS = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Courier New.ttf",
    "/System/Library/Fonts/Supplemental/Georgia.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "/System/Library/Fonts/Supplemental/Verdana.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
)

_HANDWRITTEN_FONTS = (
    "/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttf",
    "/System/Library/Fonts/Supplemental/SnellRoundhand.ttc",
    "/System/Library/Fonts/Supplemental/Apple Chancery.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
)

_ROOM_NAMES = (
    "Room",
    "Kitchen",
    "Bedroom",
    "Bathroom",
    "Hall",
    "Living",
    "Office",
    "Storage",
    "Lobby",
    "Closet",
)

_LABEL_SUFFIXES = ("A", "B", "C", "North", "South", "East", "West")


def _existing_fonts(candidates: tuple[str, ...]) -> list[str]:
    return [path for path in candidates if Path(path).is_file()]


def _load_font(path: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(path, size=size)
    except OSError:
        return ImageFont.load_default()


@dataclass(frozen=True)
class RenderedSample:
    image: Image.Image
    text: str
    width: int
    height: int


@dataclass(frozen=True)
class RenderVariations:
    """Per-effect on/off flags for render-time randomization."""

    background: bool = False
    grain: bool = False
    ink: bool = False
    handwritten: bool = False
    font: bool = False
    font_size: bool = False
    x_offset: bool = False
    line_spacing: bool = False


def _clamp_sentence(text: str, min_len: int, max_len: int) -> str:
    if len(text) > max_len:
        return text[:max_len]
    while len(text) < min_len:
        text += " "
    return text


def _random_word(rng: random.Random, min_len: int, max_len: int) -> str:
    length = rng.randint(min_len, max_len)
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(length)).capitalize()


def _random_room_label(rng: random.Random) -> str:
    base = rng.choice(_ROOM_NAMES)
    if rng.random() < 0.55:
        return f"{base} {rng.randint(100, 999)}"
    if rng.random() < 0.35:
        return f"{base} {rng.choice(_LABEL_SUFFIXES)}"
    return base


def _random_dimension(rng: random.Random) -> str:
    value = rng.uniform(1.0, 12.0)
    label = rng.choice(("Width", "Length", "Height", "Depth"))
    unit = rng.choice(("m", "cm", "ft"))
    if unit == "cm":
        return f"{label}: {value * 100:.0f}{unit}"
    return f"{label}: {value:.1f}{unit}"


def _random_identifier(rng: random.Random) -> str:
    styles = (
        lambda: f"{rng.choice(string.ascii_uppercase)}{rng.randint(1, 9)}",
        lambda: f"ID-{rng.randint(10, 9999)}",
        lambda: "".join(rng.choices(string.ascii_uppercase + string.digits, k=rng.randint(4, 8))),
    )
    return rng.choice(styles)()


def _random_line(rng: random.Random, min_len: int, max_len: int) -> str:
    generators = (
        _random_room_label,
        _random_dimension,
        _random_identifier,
        lambda r: _random_word(r, min(3, min_len), min(8, max_len)),
        lambda r: str(r.randint(0, 9999)),
        lambda r: f"{r.choice(string.ascii_uppercase)}{r.randint(1, 20)}.{r.randint(0, 9)}m",
    )
    line = rng.choice(generators)(rng)
    if rng.random() < 0.2:
        line = f"{line}{rng.choice('!.:;-')}"
    return _clamp_sentence(line, min_len, max_len)


_default_font_cache: dict[tuple[str, int], ImageFont.ImageFont] = {}


def _default_font(size: int) -> ImageFont.ImageFont:
    pool = _existing_fonts(_PRINTED_FONTS)
    path = pool[0] if pool else ""
    key = (path, size)
    cached = _default_font_cache.get(key)
    if cached is not None:
        return cached
    font = _load_font(path, size) if path else ImageFont.load_default()
    _default_font_cache[key] = font
    return font


def _pick_font(
    rng: random.Random,
    handwritten: bool,
    size: int,
    *,
    randomize: bool,
) -> ImageFont.ImageFont:
    if not randomize:
        return _default_font(size)
    pool = _existing_fonts(_HANDWRITTEN_FONTS if handwritten else _PRINTED_FONTS)
    if not pool:
        pool = _existing_fonts(_PRINTED_FONTS)
    if not pool:
        return ImageFont.load_default()
    return _load_font(rng.choice(pool), size)


def _line_bbox(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
) -> tuple[int, int, int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]


def render_text_sample(
    width: int,
    height: int,
    rng: random.Random,
    *,
    line_counts: tuple[int, ...],
    sentence_len_min: int,
    sentence_len_max: int,
    variations: RenderVariations | None = None,
) -> RenderedSample:
    """Render one grayscale image with text lines and return the transcription."""
    v = variations or RenderVariations()
    num_lines = rng.choice(line_counts)
    per_line_max = max(sentence_len_min, sentence_len_max // max(1, num_lines))
    per_line_min = min(sentence_len_min, per_line_max)
    lines = [
        _random_line(rng, per_line_min, min(sentence_len_max, per_line_max))
        for _ in range(num_lines)
    ]
    text = _clamp_sentence("\n".join(lines), sentence_len_min, sentence_len_max)

    short_side = min(width, height)
    margin = max(8, short_side // 32)
    bg_value = rng.randint(230, 255) if v.background else 255
    image = Image.new("L", (width, height), color=bg_value)
    draw = ImageDraw.Draw(image)

    y_cursor = margin
    line_gap = max(6, short_side // 40)
    default_font_size = max(18, short_side // 12)

    for line in text.split("\n"):
        handwritten = v.handwritten
        if v.font_size:
            font_size = rng.randint(max(14, short_side // 24), max(18, short_side // 10))
        else:
            font_size = default_font_size
        font = _pick_font(rng, handwritten, font_size, randomize=v.font)
        _, _, line_width, line_height = _line_bbox(draw, line, font)

        if v.x_offset:
            max_x = width - margin - line_width
            x = rng.randint(margin, max_x) if max_x >= margin else margin
        else:
            x = margin

        if y_cursor + line_height > height - margin:
            break

        ink_value = rng.randint(0, max(0, bg_value - 80)) if v.ink else 0
        draw.text((x, y_cursor), line, fill=ink_value, font=font)
        gap = rng.randint(4, line_gap) if v.line_spacing else line_gap
        y_cursor += line_height + gap

    if v.grain:
        noise = np.random.default_rng(rng.randint(0, 2**31 - 1)).integers(
            -8, 9, size=(height, width), dtype=np.int16
        )
        array = np.clip(np.asarray(image, dtype=np.int16) + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(array, mode="L")

    return RenderedSample(image=image, text=text, width=width, height=height)


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """Grayscale PIL → [1, H, W] float tensor normalized to [-1, 1]."""
    array = np.asarray(image.convert("L"), dtype=np.float32)
    tensor = torch.from_numpy(array / 127.5 - 1.0)
    return tensor.unsqueeze(0)


def pad_images_to_batch(images: list[torch.Tensor], *, pad_value: float = 1.0) -> torch.Tensor:
    max_height = max(image.shape[-2] for image in images)
    max_width = max(image.shape[-1] for image in images)
    padded: list[torch.Tensor] = []
    for image in images:
        _, height, width = image.shape
        if height == max_height and width == max_width:
            padded.append(image)
            continue
        canvas = torch.full((1, max_height, max_width), pad_value, dtype=image.dtype)
        canvas[:, :height, :width] = image
        padded.append(canvas)
    return torch.stack(padded)


class SyntheticOcrDataset(IterableDataset):
    """Infinite stream of synthetic OCR training pairs."""

    def __init__(
        self,
        *,
        image_sizes: tuple[tuple[int, int], ...],
        max_seq_len: int,
        vocab: CharVocab,
        line_counts: tuple[int, ...],
        sentence_len_min: int,
        sentence_len_max: int,
        variations: RenderVariations,
        samples_per_epoch: int,
        seed: int | None = None,
    ) -> None:
        self._image_sizes = image_sizes
        self._max_seq_len = max_seq_len
        self._vocab = vocab
        self._line_counts = line_counts
        self._sentence_len_min = sentence_len_min
        self._sentence_len_max = sentence_len_max
        self._variations = variations
        self._samples_per_epoch = samples_per_epoch
        self._seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return self._samples_per_epoch

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        rng = random.Random((self._seed or 0) + self.epoch)
        for _ in range(self._samples_per_epoch):
            width, height = rng.choice(self._image_sizes)
            sample = render_text_sample(
                width,
                height,
                rng,
                line_counts=self._line_counts,
                sentence_len_min=self._sentence_len_min,
                sentence_len_max=self._sentence_len_max,
                variations=self._variations,
            )
            token_ids = self._vocab.encode(sample.text, add_bos_eos=True)
            if len(token_ids) > self._max_seq_len:
                token_ids = token_ids[: self._max_seq_len]
                if token_ids[-1] != self._vocab.eos_id:
                    token_ids[-1] = self._vocab.eos_id
            yield image_to_tensor(sample.image), torch.tensor(token_ids, dtype=torch.long)


class OcrCollator:
    def __init__(self, pad_id: int) -> None:
        self._pad_id = pad_id

    def __call__(
        self,
        batch: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        images, sequences = zip(*batch, strict=True)
        max_len = max(seq.numel() for seq in sequences)
        padded_ids: list[torch.Tensor] = []
        for seq in sequences:
            pad_count = max_len - seq.numel()
            if pad_count:
                seq = torch.cat(
                    [seq, torch.full((pad_count,), self._pad_id, dtype=torch.long)],
                )
            padded_ids.append(seq)
        return {
            "images": pad_images_to_batch(list(images)),
            "token_ids": torch.stack(padded_ids),
        }
