"""OCR datasets: synthetic training stream and fixed holdout loader."""

from __future__ import annotations

import json
import random
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Any, Iterator

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset

from config import DataSettings, StepSchedule, VariationSettings
from renderer.corpus import WordCorpus, sample_bounded_paragraph
from renderer.paragraph_render import render_paragraph_document, render_paragraph_sample
from vocab import CharVocab


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """Grayscale PIL → [1, H, W] float tensor normalized to [-1, 1]."""
    if image.mode != "L":
        image = image.convert("L")
    array = np.asarray(image, dtype=np.float32)
    tensor = torch.from_numpy(array / 127.5 - 1.0)
    return tensor.unsqueeze(0)


class SyntheticOcrDataset(IterableDataset):
    """Infinite stream of paragraph-style OCR training pairs."""

    def __init__(
        self,
        *,
        data: DataSettings,
        corpus: WordCorpus,
        batch_size: int,
        max_seq_len: StepSchedule[int],
        vocab: CharVocab,
        samples_per_epoch: int,
        seed: int | None = None,
        training_step_value: Any = None,
        prefetch_samples: int = 0,
    ) -> None:
        self._data = data
        self._corpus = corpus
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

    def _fits_label(self, text: str, *, max_seq_len: int, max_text_chars: int) -> bool:
        if not text or len(text) > max_text_chars:
            return False
        return len(self._vocab.encode(text, add_bos_eos=True)) <= max_seq_len

    def _generate_sample(
        self,
        rng: random.Random,
        *,
        width: int,
        height: int,
        max_seq_len: int,
        variations: VariationSettings,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Render until label is non-empty drawn text with len(ids) <= max_seq_len."""
        max_text_chars = max(1, max_seq_len - 2)
        tight_budget = max_text_chars < self._data.paragraph_len_min
        max_attempts = 32

        for attempt in range(max_attempts):
            if tight_budget:
                target_max = max(1, max_text_chars - attempt // 8)
                paragraph = sample_bounded_paragraph(
                    rng,
                    self._corpus,
                    max_chars=target_max,
                    settings=variations,
                )
                sample = render_paragraph_document(
                    width,
                    height,
                    rng,
                    [paragraph],
                    variations,
                )
            else:
                shrink = min(0.75, 0.06 * attempt)
                paragraph_len_max = min(
                    max_text_chars,
                    max(
                        1,
                        int(self._data.paragraph_len_max * (1.0 - shrink)),
                    ),
                )
                paragraph_len_min = min(self._data.paragraph_len_min, paragraph_len_max)
                paragraph_counts = self._data.paragraph_counts
                if attempt >= max_attempts // 2:
                    paragraph_counts = (1,)

                sample = render_paragraph_sample(
                    width,
                    height,
                    rng,
                    self._corpus,
                    variations,
                    paragraph_counts=paragraph_counts,
                    paragraph_len_min=paragraph_len_min,
                    paragraph_len_max=paragraph_len_max,
                )

            if self._fits_label(sample.text, max_seq_len=max_seq_len, max_text_chars=max_text_chars):
                token_ids = self._vocab.encode(sample.text, add_bos_eos=True)
                return image_to_tensor(sample.image), torch.tensor(token_ids, dtype=torch.long)

        for word in sorted(self._corpus.all_words, key=len):
            for candidate in (word.capitalize() + ".", word[:max_text_chars]):
                if not candidate or len(candidate) > max_text_chars:
                    continue
                sample = render_paragraph_document(
                    width,
                    height,
                    rng,
                    [candidate],
                    variations,
                )
                if self._fits_label(
                    sample.text,
                    max_seq_len=max_seq_len,
                    max_text_chars=max_text_chars,
                ):
                    token_ids = self._vocab.encode(sample.text, add_bos_eos=True)
                    return image_to_tensor(sample.image), torch.tensor(token_ids, dtype=torch.long)

        raise RuntimeError(
            f"failed to render a sample within max_seq_len={max_seq_len} "
            f"({max_text_chars} text chars) after {max_attempts} attempts",
        )

    def _iter_samples(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        rng = random.Random((self._seed or 0) + self.epoch)
        width = height = 0
        max_seq_len = self._max_seq_len.at(0)
        variations = self._data.variations.at(0)
        for index in range(self._samples_per_epoch):
            if index % self._batch_size == 0:
                step = self._current_training_step()
                settings = self._data.at_step(step)
                width, height = rng.choice(settings.image_sizes)
                max_seq_len = self._max_seq_len.at(step)
                variations = settings.variations
            yield self._generate_sample(
                rng,
                width=width,
                height=height,
                max_seq_len=max_seq_len,
                variations=variations,
            )

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
