"""Global transformer on FPN L5 with axial 3D RoPE (y, x, pyramid level)."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

_MAX_LEVEL = 8


def _cis_1d(dim: int, max_pos: int, base: float = 10000.0) -> Tensor:
    if dim % 2 != 0:
        raise ValueError(f"RoPE axis dim ({dim}) must be even")
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    freqs = torch.outer(torch.arange(max_pos, dtype=torch.float32), inv)
    return torch.polar(torch.ones_like(freqs), freqs)


class Rotary3D(nn.Module):
    """Axial 3D RoPE over (y, x, pyramid level). head_dim must be divisible by 6."""

    def __init__(
        self,
        head_dim: int,
        *,
        max_l5_grid_side: int,
        max_level: int = _MAX_LEVEL,
    ) -> None:
        super().__init__()
        if head_dim % 6 != 0:
            raise ValueError(f"head_dim ({head_dim}) must be divisible by 6 for 3D RoPE")
        if max_l5_grid_side < 1:
            raise ValueError(f"max_l5_grid_side ({max_l5_grid_side}) must be ≥ 1")
        self.max_l5_grid_side = max_l5_grid_side
        third = head_dim // 3
        freq_y = _cis_1d(third, max_l5_grid_side)
        freq_x = _cis_1d(third, max_l5_grid_side)
        freq_z = _cis_1d(third, max_level)
        cis = torch.cat(
            (
                freq_y[:, None, None, :].expand(max_l5_grid_side, max_l5_grid_side, max_level, -1),
                freq_x[None, :, None, :].expand(max_l5_grid_side, max_l5_grid_side, max_level, -1),
                freq_z[None, None, :, :].expand(max_l5_grid_side, max_l5_grid_side, max_level, -1),
            ),
            dim=-1,
        )
        self.register_buffer("cis", cis, persistent=False)

    def slice(self, height: int, width: int, level: int) -> Tensor:
        if height > self.max_l5_grid_side or width > self.max_l5_grid_side:
            raise ValueError(
                f"L5 grid ({height}×{width}) exceeds max_l5_grid_side ({self.max_l5_grid_side})"
            )
        if level < 0 or level >= _MAX_LEVEL:
            raise ValueError(f"pyramid level ({level}) must be in [0, {_MAX_LEVEL})")
        return self.cis[:height, :width, level].reshape(height * width, -1)


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


class L5Transformer(nn.Module):
    """Self-attention on L5 tokens after lateral projection."""

    def __init__(
        self,
        *,
        in_channels: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_ratio: float,
        level: int,
        max_l5_grid_side: int,
    ) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by heads ({heads})")
        head_dim = dim // heads
        if head_dim % 6 != 0:
            raise ValueError(
                f"head_dim ({head_dim}) must be divisible by 6 for 3D RoPE; "
                f"adjust dim or heads"
            )
        self.level = level
        self.in_proj = (
            nn.Conv2d(in_channels, dim, kernel_size=1)
            if in_channels != dim
            else nn.Identity()
        )
        self.rope = Rotary3D(head_dim, max_l5_grid_side=max_l5_grid_side)
        self.blocks = nn.ModuleList(
            TransformerBlock(dim, heads, mlp_ratio) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(dim)
        self.out_proj = (
            nn.Conv2d(dim, in_channels, kernel_size=1)
            if in_channels != dim
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        b, _, height, width = x.shape
        h = self.in_proj(x)
        tokens = h.flatten(2).transpose(1, 2)
        cis = self.rope.slice(height, width, self.level)
        for block in self.blocks:
            tokens = block(tokens, cis)
        tokens = self.norm(tokens)
        h = tokens.transpose(1, 2).view(b, -1, height, width)
        return x + self.out_proj(h)
