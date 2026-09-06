"""Hub image → taef2 latents; simple_image latent reused for identity pairs."""

from __future__ import annotations

import time
import urllib.request
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from PIL import Image
from registry import load_hub_dataset
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import IterableDataset

from taesd import TAESD

_HUB_RETRIES = 5
_SHUFFLE_BUFFER = 256
_TAESD_SCALE = 8
_ENCODER_URL = "https://github.com/madebyollin/taesd/raw/main/taef2_encoder.pth"
_DECODER_URL = "https://github.com/madebyollin/taesd/raw/main/taef2_decoder.pth"


def taesd_hw(image_size: int) -> int:
    return image_size // _TAESD_SCALE


def prepare_image(image: Image.Image | np.ndarray, size: int) -> torch.Tensor:
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    image = image.convert("L").resize((size, size), Image.Resampling.BILINEAR)
    array = np.clip(np.asarray(image, dtype=np.float32) / 255.0, 0.0, 1.0)
    return torch.from_numpy(array).unsqueeze(0)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {dest.name} …", flush=True)
    urllib.request.urlretrieve(url, dest)


def ensure_taesd_weights(encoder: Path, decoder: Path) -> None:
    if not encoder.exists():
        _download(_ENCODER_URL, encoder)
    if not decoder.exists():
        _download(_DECODER_URL, decoder)


def build_taesd(encoder_path: Path, decoder_path: Path, device: torch.device) -> TAESD:
    ensure_taesd_weights(encoder_path, decoder_path)
    taesd = TAESD(encoder_path=str(encoder_path), decoder_path=str(decoder_path))
    for param in taesd.parameters():
        param.requires_grad = False
    taesd.eval()
    return taesd.to(device)


def encode_image(taesd: TAESD, x: Tensor) -> Tensor:
    rgb = x.expand(-1, 3, -1, -1) if x.shape[1] == 1 else x
    with torch.no_grad():
        return taesd.encoder(rgb)


def decode_latent(taesd: TAESD, z: Tensor) -> Tensor:
    with torch.no_grad():
        return taesd.decoder(z).clamp(0, 1).mean(dim=1, keepdim=True)


def latent_ink_weight(target_image: Tensor, *, ink_weight: float) -> Tensor:
    """Downsample grayscale target to taef2 spatial size for ink-weighted latent loss."""
    weight = 1.0 + ink_weight * (1.0 - target_image)
    return F.avg_pool2d(weight, kernel_size=_TAESD_SCALE, stride=_TAESD_SCALE)


def _is_timeout(exc: BaseException) -> bool:
    return "timeout" in type(exc).__name__.lower() or "timed out" in str(exc).lower()


def load_rows(dataset: str, *, revision: str | None = None, cache_dir: Path | None = None):
    for attempt in range(1, _HUB_RETRIES + 1):
        try:
            return load_hub_dataset(dataset, revision=revision, cache_dir=cache_dir)
        except Exception as exc:
            if not _is_timeout(exc) or attempt == _HUB_RETRIES:
                raise
            wait = min(2 ** (attempt - 1), 16)
            print(f"Hub timeout ({attempt}/{_HUB_RETRIES}), retry in {wait}s…", flush=True)
            time.sleep(wait)


def hub_row_count(rows: Any) -> int:
    splits = getattr(rows.info, "splits", None) or {}
    train = splits.get("train") if hasattr(splits, "get") else None
    if train is None or not train.num_examples:
        raise FileNotFoundError("Hub stream has no train example count")
    return int(train.num_examples)


class FloorplanPairDataset(IterableDataset):
    def __init__(
        self,
        rows: Any,
        image_size: int,
        *,
        num_rows: int,
        precompute_latent: bool = True,
        taesd: TAESD | None = None,
        device: torch.device | None = None,
        ink_weight: float = 10.0,
        seed: int | None = None,
    ) -> None:
        if precompute_latent and (taesd is None or device is None):
            raise ValueError("taesd and device are required when precompute_latent is true")
        self._rows = rows
        self._image_size = image_size
        self._precompute_latent = precompute_latent
        self._taesd = taesd
        self._device = device
        self._num_rows = num_rows
        self._ink_weight = ink_weight
        self._seed = seed or 0
        self.epoch = 0

    def __len__(self) -> int:
        return self._num_rows * 2

    def _encode(self, image: torch.Tensor) -> torch.Tensor:
        batch = image.unsqueeze(0).to(self._device)
        return encode_image(self._taesd, batch).squeeze(0).cpu()

    def __iter__(self) -> Iterator[tuple[torch.Tensor, ...]]:
        stream = self._rows.shuffle(seed=self._seed + self.epoch, buffer_size=_SHUFFLE_BUFFER)
        for row in stream:
            source_image = prepare_image(row["image"], self._image_size)
            target_image = prepare_image(row["simple_image"], self._image_size)
            if not self._precompute_latent:
                yield source_image, target_image
                yield target_image, target_image
                continue
            source_latent = self._encode(source_image)
            target_latent = self._encode(target_image)
            weight = latent_ink_weight(target_image, ink_weight=self._ink_weight)
            yield source_latent, target_latent, weight, source_image, target_image
            yield target_latent, target_latent, weight, target_image, target_image
