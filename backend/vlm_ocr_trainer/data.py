"""On-the-fly synthetic OCR samples: random text lines → grayscale image."""

from __future__ import annotations

import json
import random
import string
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Any, Iterator

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset, IterableDataset

from config import DataSettings, RenderVariations
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


_PRINTED_FONT_PATHS = _existing_fonts(_PRINTED_FONTS)
_HANDWRITTEN_FONT_PATHS = _existing_fonts(_HANDWRITTEN_FONTS)
_FONT_CACHE: dict[tuple[str, int], ImageFont.ImageFont] = {}


def _load_font(path: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    key = (path, size)
    cached = _FONT_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        font = ImageFont.truetype(path, size=size)
    except OSError:
        font = ImageFont.load_default()
    _FONT_CACHE[key] = font
    return font


@dataclass(frozen=True)
class RenderedSample:
    image: Image.Image
    text: str
    width: int
    height: int


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


def _default_font(size: int) -> ImageFont.ImageFont:
    path = _PRINTED_FONT_PATHS[0] if _PRINTED_FONT_PATHS else ""
    return _load_font(path, size) if path else ImageFont.load_default()


def _pick_font(
    rng: random.Random,
    handwritten: bool,
    size: int,
    *,
    randomize: bool,
) -> ImageFont.ImageFont:
    if not randomize:
        return _default_font(size)
    pool = _HANDWRITTEN_FONT_PATHS if handwritten else _PRINTED_FONT_PATHS
    if not pool:
        pool = _PRINTED_FONT_PATHS
    if not pool:
        return ImageFont.load_default()
    return _load_font(rng.choice(pool), size)


def _text_width(font: ImageFont.ImageFont, text: str) -> float:
    getlength = getattr(font, "getlength", None)
    if getlength is not None:
        return float(getlength(text))
    if hasattr(font, "getbbox"):
        bbox = font.getbbox(text)
        return float(bbox[2] - bbox[0])
    return float(len(text) * 8)


def _text_bbox(font: ImageFont.ImageFont, text: str) -> tuple[int, int, int, int]:
    if hasattr(font, "getbbox"):
        left, top, right, bottom = font.getbbox(text)
        return left, top, right - left, bottom - top
    width = int(_text_width(font, text))
    return 0, 0, width, 12


def _wrap_long_word(word: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    rows: list[str] = []
    row = ""
    for character in word:
        candidate = row + character
        if row and _text_width(font, candidate) > max_width:
            rows.append(row)
            row = character
        else:
            row = candidate
    if row:
        rows.append(row)
    return rows


def _draw_text_row(
    image: Image.Image,
    *,
    row: str,
    font: ImageFont.ImageFont,
    x: int,
    y: int,
    ink_value: int,
    bg_value: int,
    tilt: bool,
    rng: random.Random,
) -> None:
    if not tilt:
        ImageDraw.Draw(image).text((x, y), row, fill=ink_value, font=font)
        return

    left, top, right, bottom = font.getbbox(row)
    row_width = right - left
    row_height = bottom - top
    pad = max(8, row_height // 2)
    line_img = Image.new("L", (row_width + 2 * pad, row_height + 2 * pad), color=bg_value)
    ImageDraw.Draw(line_img).text((pad - left, pad - top), row, fill=ink_value, font=font)
    angle = rng.uniform(-4.0, 4.0)
    rotated = line_img.rotate(
        angle,
        center=(pad, pad),
        expand=True,
        resample=Image.Resampling.BICUBIC,
        fillcolor=bg_value,
    )
    image.paste(rotated, (x - pad, y - pad))


def _wrap_line(text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    if not text:
        return []
    rows: list[str] = []
    for word in text.split(" "):
        if not word:
            continue
        if _text_width(font, word) > max_width:
            if rows:
                rows.extend(_wrap_long_word(word, font, max_width))
            else:
                rows = _wrap_long_word(word, font, max_width)
            continue
        if not rows:
            rows.append(word)
            continue
        candidate = f"{rows[-1]} {word}"
        if _text_width(font, candidate) > max_width:
            rows.append(word)
        else:
            rows[-1] = candidate
    return rows


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

    y_cursor = margin
    line_gap = max(6, short_side // 40)
    default_font_size = max(18, short_side // 12)
    rendered_rows: list[str] = []
    canvas_full = False

    for line in text.split("\n"):
        handwritten = v.handwritten
        if v.font_size:
            font_size = rng.randint(max(14, short_side // 24), max(18, short_side // 10))
        else:
            font_size = default_font_size
        font = _pick_font(rng, handwritten, font_size, randomize=v.font)
        rows = _wrap_line(line.rstrip(), font, width - 2 * margin)
        for row in rows:
            _, _, row_width, row_height = _text_bbox(font, row)
            if y_cursor + row_height > height - margin:
                canvas_full = True
                break
            if v.x_offset:
                max_x = width - margin - row_width
                x = rng.randint(margin, max_x) if max_x >= margin else margin
            else:
                x = margin

            ink_value = rng.randint(0, max(0, bg_value - 80)) if v.ink else 0
            _draw_text_row(
                image,
                row=row,
                font=font,
                x=x,
                y=y_cursor,
                ink_value=ink_value,
                bg_value=bg_value,
                tilt=v.line_tilt,
                rng=rng,
            )
            rendered_rows.append(row)
            gap = rng.randint(4, line_gap) if v.line_spacing else line_gap
            y_cursor += row_height + gap
        if canvas_full:
            break

    text = "\n".join(rendered_rows)

    if v.grain:
        noise = np.random.default_rng(rng.randint(0, 2**31 - 1)).integers(
            -8, 9, size=(height, width), dtype=np.int16
        )
        array = np.clip(np.asarray(image, dtype=np.int16) + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(array, mode="L")

    return RenderedSample(image=image, text=text, width=width, height=height)


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """Grayscale PIL → [1, H, W] float tensor normalized to [-1, 1]."""
    if image.mode != "L":
        image = image.convert("L")
    array = np.asarray(image, dtype=np.float32)
    tensor = torch.from_numpy(array / 127.5 - 1.0)
    return tensor.unsqueeze(0)


class SyntheticOcrDataset(IterableDataset):
    """Infinite stream of synthetic OCR training pairs."""

    def __init__(
        self,
        *,
        data: DataSettings,
        batch_size: int,
        max_seq_len: int,
        vocab: CharVocab,
        samples_per_epoch: int,
        seed: int | None = None,
        training_step_value: Any = None,
        prefetch_samples: int = 0,
    ) -> None:
        self._data = data
        self._batch_size = max(1, batch_size)
        self._max_seq_len = max_seq_len
        self._vocab = vocab
        self._samples_per_epoch = samples_per_epoch
        self._seed = seed
        self._training_step_value = training_step_value
        self._prefetch_samples = max(0, prefetch_samples)
        self.epoch = 0
        self._local_training_step = 0

    @property
    def training_step(self) -> int:
        return self._local_training_step

    @training_step.setter
    def training_step(self, step: int) -> None:
        self._local_training_step = step
        if self._training_step_value is not None:
            self._training_step_value.value = step

    def _current_training_step(self) -> int:
        if self._training_step_value is not None:
            return int(self._training_step_value.value)
        return self._local_training_step

    def __len__(self) -> int:
        return self._samples_per_epoch

    def _generate_sample(
        self,
        rng: random.Random,
        *,
        width: int,
        height: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        settings = self._data.at_step(self._current_training_step())
        sample = render_text_sample(
            width,
            height,
            rng,
            line_counts=settings.line_counts,
            sentence_len_min=settings.sentence_len_min,
            sentence_len_max=settings.sentence_len_max,
            variations=settings.variations,
        )
        token_ids = self._vocab.encode(sample.text, add_bos_eos=True)
        if len(token_ids) > self._max_seq_len:
            token_ids = token_ids[: self._max_seq_len]
            if token_ids[-1] != self._vocab.eos_id:
                token_ids[-1] = self._vocab.eos_id
        return image_to_tensor(sample.image), torch.tensor(token_ids, dtype=torch.long)

    def _iter_samples(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        rng = random.Random((self._seed or 0) + self.epoch)
        width = height = 0
        for index in range(self._samples_per_epoch):
            if index % self._batch_size == 0:
                settings = self._data.at_step(self._current_training_step())
                width, height = rng.choice(settings.image_sizes)
            yield self._generate_sample(rng, width=width, height=height)

    def _iter_prefetched(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        queue: Queue[tuple[torch.Tensor, torch.Tensor] | None] = Queue(
            maxsize=self._prefetch_samples,
        )

        def producer() -> None:
            try:
                for item in self._iter_samples():
                    queue.put(item)
            finally:
                queue.put(None)

        thread = Thread(target=producer, daemon=True)
        thread.start()
        while True:
            try:
                item = queue.get(timeout=0.1)
            except Empty:
                if not thread.is_alive() and queue.empty():
                    break
                continue
            if item is None:
                break
            yield item

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        if self._prefetch_samples > 0:
            yield from self._iter_prefetched()
            return
        yield from self._iter_samples()


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
            "images": torch.stack(images),
            "token_ids": torch.stack(padded_ids),
        }


class HoldoutDataset(Dataset):
    """Fixed holdout samples loaded from manifest.jsonl."""

    def __init__(self, manifest_path: Path, holdout_dir: Path) -> None:
        self._holdout_dir = holdout_dir
        self._records: list[dict[str, object]] = []
        with manifest_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    self._records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        record = self._records[index]
        image_path = self._holdout_dir / str(record["image"])
        image = Image.open(image_path).convert("L")
        return image_to_tensor(image), str(record["text"])


class HoldoutCollator:
    def __call__(
        self,
        batch: list[tuple[torch.Tensor, str]],
    ) -> dict[str, torch.Tensor | list[str]]:
        images, texts = zip(*batch, strict=True)
        shapes = {tuple(image.shape) for image in images}
        if len(shapes) != 1:
            raise ValueError("holdout batch contains mixed image sizes; match benchmark batch_size")
        return {
            "images": torch.stack(images),
            "texts": list(texts),
        }
