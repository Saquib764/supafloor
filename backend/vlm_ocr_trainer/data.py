"""OCR datasets: synthetic training stream and fixed holdout loader."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Any, Iterator

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset

from config import DataSettings, StepSchedule, SeqLenMix, sample_max_seq_len
from renderer.corpus import WordCorpus
from renderer.training_sample import generate_training_sample
from vocab import CharVocab


def _batch_plan_rng(seed: int, epoch: int, global_batch_id: int) -> random.Random:
    return random.Random(seed + epoch * 1_000_003 + global_batch_id * 9_171_231)


def _content_rng(seed: int, epoch: int, worker_id: int, index: int) -> random.Random:
    return random.Random(seed + epoch * 1_000_003 + worker_id * 1_000_009 + index * 7919)


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
        max_seq_len: StepSchedule[SeqLenMix],
        vocab: CharVocab,
        samples_per_epoch: int,
        seed: int | None = None,
        training_step_value: Any = None,
        epoch_value: Any = None,
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
        self._epoch_value = epoch_value
        self._prefetch_samples = max(0, prefetch_samples)
        self._local_epoch = 0
        self._local_training_step = 0
        self._last_batch_data_sec = 0.0

    @property
    def epoch(self) -> int:
        return self._current_epoch()

    @epoch.setter
    def epoch(self, value: int) -> None:
        self._local_epoch = value
        if self._epoch_value is not None:
            self._epoch_value.value = value

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

    def _current_epoch(self) -> int:
        if self._epoch_value is not None:
            return int(self._epoch_value.value)
        return self._local_epoch

    def _worker_batch_shard(self) -> tuple[int, int, int]:
        info = torch.utils.data.get_worker_info()
        batches_per_epoch = self._samples_per_epoch // self._batch_size
        if info is None:
            return 0, 1, batches_per_epoch
        per_worker = batches_per_epoch // info.num_workers
        extra = batches_per_epoch % info.num_workers
        count = per_worker + (1 if info.id < extra else 0)
        return info.id, info.num_workers, count

    def __len__(self) -> int:
        return self._samples_per_epoch

    @property
    def last_batch_data_sec(self) -> float:
        return self._last_batch_data_sec

    def _generate_sample(
        self,
        rng: random.Random,
        *,
        width: int,
        height: int,
        max_seq_len: int,
        variations,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sample = generate_training_sample(
            width,
            height,
            rng,
            self._corpus,
            variations,
            self._vocab,
            max_seq_len=max_seq_len,
            paragraph_counts=self._data.paragraph_counts,
        )
        token_ids = self._vocab.encode(sample.text, add_bos_eos=True)
        return image_to_tensor(sample.image), torch.tensor(token_ids, dtype=torch.long)

    def _build_batch(self, global_batch_id: int, worker_id: int) -> dict[str, torch.Tensor]:
        seed = self._seed or 0
        epoch = self._current_epoch()
        step = self._current_training_step()
        settings = self._data.at_step(step)
        plan_rng = _batch_plan_rng(seed, epoch, global_batch_id)
        width, height = plan_rng.choice(settings.image_sizes)
        max_seq_len = sample_max_seq_len(self._max_seq_len.at(step), plan_rng)
        variations = settings.variations
        samples = [
            self._generate_sample(
                _content_rng(seed, epoch, worker_id, global_batch_id * self._batch_size + sample_idx),
                width=width,
                height=height,
                max_seq_len=max_seq_len,
                variations=variations,
            )
            for sample_idx in range(self._batch_size)
        ]
        return OcrCollator(self._vocab.pad_id)(samples)

    def _iter_batches(self) -> Iterator[dict[str, torch.Tensor]]:
        worker_id, num_workers, batch_count = self._worker_batch_shard()
        for local_batch in range(batch_count):
            global_batch_id = local_batch * num_workers + worker_id
            t0 = time.perf_counter()
            yield self._build_batch(global_batch_id, worker_id)
            self._last_batch_data_sec = time.perf_counter() - t0

    def _iter_prefetched(self) -> Iterator[dict[str, torch.Tensor]]:
        queue: Queue[dict[str, torch.Tensor] | None] = Queue(
            maxsize=max(1, self._prefetch_samples // self._batch_size),
        )

        def producer() -> None:
            try:
                for item in self._iter_batches():
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

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        if self._prefetch_samples > 0:
            yield from self._iter_prefetched()
            return
        yield from self._iter_batches()


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


class OcrBatchCollator:
    """Unwrap pre-built batches from SyntheticOcrDataset (DataLoader batch_size=1)."""

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return batch[0]


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
            raise ValueError("holdout batch contains mixed image sizes; match benchmark.batch_size")
        return {
            "images": torch.stack(images),
            "texts": list(texts),
        }
