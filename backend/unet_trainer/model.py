"""Frozen taef2 + compressor + ViT + expander: 1-channel image → simple_image."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from config import Config
from taesd import TAESD

_IMAGE_SIZE_MIN = 512
_IMAGE_SIZE_MAX = 1024
_TAESD_SCALE = 8
_COMPRESS_SCALE = 2
_TAESD_CHANNELS = 32
_BOTTLENECK_CHANNELS = 16
_ENCODER_URL = "https://github.com/madebyollin/taesd/raw/main/taef2_encoder.pth"
_DECODER_URL = "https://github.com/madebyollin/taesd/raw/main/taef2_decoder.pth"
_TRAINABLE = frozenset({"compressor", "vit", "expander"})


def compressed_hw(image_size: int, patch_size: int) -> int:
    if patch_size < 1:
        raise ValueError(f"vit.patch_size ({patch_size}) must be ≥ 1")
    stride = _TAESD_SCALE * _COMPRESS_SCALE * patch_size
    if image_size < _IMAGE_SIZE_MIN or image_size > _IMAGE_SIZE_MAX:
        raise ValueError(
            f"image_size ({image_size}) must be in {_IMAGE_SIZE_MIN}–{_IMAGE_SIZE_MAX}"
        )
    if image_size % stride != 0:
        raise ValueError(
            f"image_size ({image_size}) must be divisible by {stride} "
            f"(8× TAESD × 2× compress × vit.patch_size {patch_size})"
        )
    return image_size // (_TAESD_SCALE * _COMPRESS_SCALE)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {dest.name} …", flush=True)
    urllib.request.urlretrieve(url, dest)


def ensure_taesd_weights(encoder: Path, decoder: Path) -> None:
    if not encoder.exists():
        _download(_ENCODER_URL, encoder)
    if not decoder.exists():
        _download(_DECODER_URL, decoder)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = min(8, channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class Compressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(_TAESD_CHANNELS, _BOTTLENECK_CHANNELS, 3, stride=2, padding=1)
        self.norm = nn.GroupNorm(min(8, _BOTTLENECK_CHANNELS), _BOTTLENECK_CHANNELS)
        self.res = ResidualBlock(_BOTTLENECK_CHANNELS)

    def forward(self, x: Tensor) -> Tensor:
        return self.res(F.silu(self.norm(self.conv(x))))


class Expander(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(_BOTTLENECK_CHANNELS, _TAESD_CHANNELS, 4, stride=2, padding=1)
        self.norm = nn.GroupNorm(min(8, _TAESD_CHANNELS), _TAESD_CHANNELS)
        self.res = ResidualBlock(_TAESD_CHANNELS)

    def forward(self, x: Tensor) -> Tensor:
        return self.res(F.silu(self.norm(self.up(x))))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        return x + self.mlp(self.norm2(x))


class LatentViT(nn.Module):
    def __init__(
        self,
        *,
        spatial: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_ratio: float,
        patch_size: int,
    ) -> None:
        super().__init__()
        if spatial % patch_size != 0:
            raise ValueError(f"compressed spatial ({spatial}) must be divisible by patch_size ({patch_size})")
        self.patch_size = patch_size
        self.grid = spatial // patch_size
        n_tokens = self.grid * self.grid
        self.embed = nn.Conv2d(_BOTTLENECK_CHANNELS, dim, kernel_size=patch_size, stride=patch_size)
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, dim))
        self.blocks = nn.ModuleList(TransformerBlock(dim, heads, mlp_ratio) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, _BOTTLENECK_CHANNELS * patch_size * patch_size)

    def forward(self, x: Tensor) -> Tensor:
        b, _, height, width = x.shape
        tokens = self.embed(x).flatten(2).transpose(1, 2)
        tokens = tokens + self.pos
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.proj(self.norm(tokens))
        patch = self.patch_size
        tokens = tokens.view(b, self.grid, self.grid, _BOTTLENECK_CHANNELS, patch, patch)
        tokens = tokens.permute(0, 3, 1, 4, 2, 5).contiguous()
        return x + tokens.view(b, _BOTTLENECK_CHANNELS, height, width)


class FloorplanLatentViT(nn.Module):
    def __init__(
        self,
        *,
        image_size: int,
        encoder_path: Path,
        decoder_path: Path,
        vit_dim: int,
        vit_depth: int,
        vit_heads: int,
        vit_mlp_ratio: float,
        vit_patch_size: int,
    ) -> None:
        super().__init__()
        spatial = compressed_hw(image_size, vit_patch_size)
        taesd = TAESD(encoder_path=str(encoder_path), decoder_path=str(decoder_path))
        for param in taesd.parameters():
            param.requires_grad = False
        taesd.eval()
        self.taesd = taesd
        self.compressor = Compressor()
        self.vit = LatentViT(
            spatial=spatial,
            dim=vit_dim,
            depth=vit_depth,
            heads=vit_heads,
            mlp_ratio=vit_mlp_ratio,
            patch_size=vit_patch_size,
        )
        self.expander = Expander()

    def train(self, mode: bool = True):
        super().train(mode)
        self.taesd.eval()
        return self

    def encode_taesd(self, x: Tensor) -> Tensor:
        rgb = x.expand(-1, 3, -1, -1) if x.shape[1] == 1 else x
        with torch.no_grad():
            return self.taesd.encoder(rgb)

    def decode_taesd(self, z: Tensor) -> Tensor:
        return self.taesd.decoder(z).clamp(0, 1).mean(dim=1, keepdim=True)

    def forward(self, x: Tensor) -> Tensor:
        z = self.encode_taesd(x)
        z = self.expander(self.vit(self.compressor(z)))
        return self.decode_taesd(z)

    def trainable_state_dict(self) -> dict[str, dict[str, Tensor]]:
        return {name: getattr(self, name).state_dict() for name in _TRAINABLE}

    def load_trainable_state_dict(self, state: dict[str, dict[str, Tensor]]) -> None:
        for name in _TRAINABLE:
            getattr(self, name).load_state_dict(state[name])


def build_model(cfg: Config) -> FloorplanLatentViT:
    ensure_taesd_weights(cfg.taesd.encoder, cfg.taesd.decoder)
    return FloorplanLatentViT(
        image_size=cfg.image_size,
        encoder_path=cfg.taesd.encoder,
        decoder_path=cfg.taesd.decoder,
        vit_dim=cfg.vit.dim,
        vit_depth=cfg.vit.depth,
        vit_heads=cfg.vit.heads,
        vit_mlp_ratio=cfg.vit.mlp_ratio,
        vit_patch_size=cfg.vit.patch_size,
    )


def recon_loss(recon: Tensor, target: Tensor, *, ink_weight: float) -> Tensor:
    weight = 1.0 + ink_weight * (1.0 - target)
    return (weight * (recon - target).abs()).mean()
