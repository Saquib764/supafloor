"""Compressor + ViT + expander in taef2 latent space (optional taef2 wrap for image I/O)."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from config import Config
from data import ensure_taesd_weights
from taesd import TAESD

_IMAGE_SIZE_MIN = 256
_IMAGE_SIZE_MAX = 1024
_TAESD_SCALE = 8
_COMPRESS_SCALE = 2
_TAESD_CHANNELS = 32
_BOTTLENECK_CHANNELS = 16
_MAX_GRID = _IMAGE_SIZE_MAX // (_TAESD_SCALE * _COMPRESS_SCALE)
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


def _cis_1d(dim: int, max_pos: int, base: float = 10000.0) -> Tensor:
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    freqs = torch.outer(torch.arange(max_pos, dtype=torch.float32), inv)
    return torch.polar(torch.ones_like(freqs), freqs)


class Rotary2D(nn.Module):
    """Axial 2D RoPE table for compressed grids up to _MAX_GRID × _MAX_GRID."""

    def __init__(self, head_dim: int, max_grid: int = _MAX_GRID) -> None:
        super().__init__()
        if head_dim % 4 != 0:
            raise ValueError(f"head_dim ({head_dim}) must be divisible by 4 for 2D RoPE")
        half = head_dim // 2
        freq_y = _cis_1d(half, max_grid)
        freq_x = _cis_1d(half, max_grid)
        cis = torch.cat(
            (
                freq_y[:, None, :].expand(max_grid, max_grid, -1),
                freq_x[None, :, :].expand(max_grid, max_grid, -1),
            ),
            dim=-1,
        )
        self.register_buffer("cis", cis, persistent=False)

    def slice(self, height: int, width: int) -> Tensor:
        if height > _MAX_GRID or width > _MAX_GRID:
            raise ValueError(
                f"grid ({height}×{width}) exceeds max RoPE grid ({_MAX_GRID}×{_MAX_GRID})"
            )
        return self.cis[:height, :width].reshape(height * width, -1)


def apply_rope(x: Tensor, cis: Tensor) -> Tensor:
    x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(x_c * cis).flatten(-2).type_as(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by heads ({heads})")
        self.heads = heads
        self.head_dim = dim // heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: Tensor, cis: Tensor) -> Tensor:
        b, tokens, _ = x.shape
        h = self.norm1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(b, tokens, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(b, tokens, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(b, tokens, self.heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cis)
        k = apply_rope(k, cis)
        h = F.scaled_dot_product_attention(q, k, v)
        h = h.transpose(1, 2).reshape(b, tokens, -1)
        x = x + self.out_proj(h)
        return x + self.mlp(self.norm2(x))


class LatentViT(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        depth: int,
        heads: int,
        mlp_ratio: float,
        patch_size: int,
    ) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by heads ({heads})")
        self.patch_size = patch_size
        self.embed = nn.Conv2d(_BOTTLENECK_CHANNELS, dim, kernel_size=patch_size, stride=patch_size)
        self.rope = Rotary2D(dim // heads)
        self.blocks = nn.ModuleList(TransformerBlock(dim, heads, mlp_ratio) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, _BOTTLENECK_CHANNELS * patch_size * patch_size)

    def forward(self, x: Tensor) -> Tensor:
        b, _, height, width = x.shape
        tokens = self.embed(x)
        _, _, grid_h, grid_w = tokens.shape
        tokens = tokens.flatten(2).transpose(1, 2)
        cis = self.rope.slice(grid_h, grid_w)
        for block in self.blocks:
            tokens = block(tokens, cis)
        tokens = self.proj(self.norm(tokens))
        patch = self.patch_size
        tokens = tokens.view(b, grid_h, grid_w, _BOTTLENECK_CHANNELS, patch, patch)
        tokens = tokens.permute(0, 3, 1, 4, 2, 5).contiguous()
        return x + tokens.view(b, _BOTTLENECK_CHANNELS, height, width)


class FloorplanLatentViT(nn.Module):
    def __init__(
        self,
        *,
        image_size: int,
        vit_dim: int,
        vit_depth: int,
        vit_heads: int,
        vit_mlp_ratio: float,
        vit_patch_size: int,
        precompute_latent: bool = True,
        encoder_path: Path | None = None,
        decoder_path: Path | None = None,
    ) -> None:
        super().__init__()
        compressed_hw(image_size, vit_patch_size)
        self.precompute_latent = precompute_latent
        self.taesd = None
        if not precompute_latent:
            if encoder_path is None or decoder_path is None:
                raise ValueError("encoder_path and decoder_path required when precompute_latent is false")
            ensure_taesd_weights(encoder_path, decoder_path)
            taesd = TAESD(encoder_path=str(encoder_path), decoder_path=str(decoder_path))
            for param in taesd.parameters():
                param.requires_grad = False
            taesd.eval()
            self.taesd = taesd
        self.compressor = Compressor()
        self.vit = LatentViT(
            dim=vit_dim,
            depth=vit_depth,
            heads=vit_heads,
            mlp_ratio=vit_mlp_ratio,
            patch_size=vit_patch_size,
        )
        self.expander = Expander()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.taesd is not None:
            self.taesd.eval()
        return self

    def encode_taesd(self, x: Tensor) -> Tensor:
        if self.taesd is None:
            raise RuntimeError("encode_taesd requires taesd (precompute_latent=false)")
        rgb = x.expand(-1, 3, -1, -1) if x.shape[1] == 1 else x
        with torch.no_grad():
            return self.taesd.encoder(rgb)

    def decode_taesd(self, z: Tensor) -> Tensor:
        if self.taesd is None:
            raise RuntimeError("decode_taesd requires taesd (precompute_latent=false)")
        return self.taesd.decoder(z).clamp(0, 1).mean(dim=1, keepdim=True)

    def forward_latent(self, z: Tensor) -> Tensor:
        return self.expander(self.vit(self.compressor(z)))

    def forward(self, x: Tensor) -> Tensor:
        if self.taesd is not None:
            z = self.encode_taesd(x)
            z = self.forward_latent(z)
            return self.decode_taesd(z)
        return self.forward_latent(x)

    def trainable_state_dict(self) -> dict[str, dict[str, Tensor]]:
        return {name: getattr(self, name).state_dict() for name in _TRAINABLE}

    def load_trainable_state_dict(self, state: dict[str, dict[str, Tensor]]) -> None:
        for name in _TRAINABLE:
            getattr(self, name).load_state_dict(state[name])


def build_model(cfg: Config) -> FloorplanLatentViT:
    precompute = cfg.train.precompute_latent
    return FloorplanLatentViT(
        image_size=cfg.image_size,
        vit_dim=cfg.vit.dim,
        vit_depth=cfg.vit.depth,
        vit_heads=cfg.vit.heads,
        vit_mlp_ratio=cfg.vit.mlp_ratio,
        vit_patch_size=cfg.vit.patch_size,
        precompute_latent=precompute,
        encoder_path=cfg.taesd.encoder if not precompute else None,
        decoder_path=cfg.taesd.decoder if not precompute else None,
    )


def latent_recon_loss(recon: Tensor, target: Tensor, *, weight: Tensor) -> Tensor:
    diff = (recon - target).abs()
    return (weight * diff).mean()


def image_recon_loss(recon: Tensor, target: Tensor, *, ink_weight: float) -> Tensor:
    weight = 1.0 + ink_weight * (1.0 - target)
    return (weight * (recon - target).abs()).mean()
