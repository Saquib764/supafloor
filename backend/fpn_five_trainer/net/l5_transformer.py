"""Global transformer on FPN L5 with axial 2D RoPE (y, x)."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _inv_freq(dim: int, base: float = 10000.0) -> Tensor:
    if dim % 2 != 0:
        raise ValueError(f"RoPE axis dim ({dim}) must be even")
    return 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))


def _cis_from_positions(positions: Tensor, inv_freq: Tensor) -> Tensor:
    freqs = positions.unsqueeze(-1) * inv_freq.unsqueeze(0)
    return torch.polar(torch.ones_like(freqs), freqs)


def _normalized_axis(length: int, *, device: torch.device) -> Tensor:
    if length <= 1:
        return torch.zeros(1, device=device, dtype=torch.float32)
    return torch.arange(length, device=device, dtype=torch.float32) / float(length - 1)


class Rotary2D(nn.Module):
    """Axial 2D RoPE with grid coords normalized to [0, 1] via i / (L - 1)."""

    def __init__(self, head_dim: int, *, max_l5_grid_side: int) -> None:
        super().__init__()
        if head_dim % 4 != 0:
            raise ValueError(f"head_dim ({head_dim}) must be divisible by 4 for 2D RoPE")
        if max_l5_grid_side < 1:
            raise ValueError(f"max_l5_grid_side ({max_l5_grid_side}) must be ≥ 1")
        self.max_l5_grid_side = max_l5_grid_side
        self.head_dim = head_dim
        axis_dim = head_dim // 2
        self.register_buffer("inv_freq_y", _inv_freq(axis_dim), persistent=False)
        self.register_buffer("inv_freq_x", _inv_freq(axis_dim), persistent=False)

    def slice(self, height: int, width: int) -> Tensor:
        if height > self.max_l5_grid_side or width > self.max_l5_grid_side:
            raise ValueError(
                f"L5 grid ({height}×{width}) exceeds max_l5_grid_side ({self.max_l5_grid_side})"
            )
        device = self.inv_freq_y.device
        y_pos = _normalized_axis(height, device=device)
        x_pos = _normalized_axis(width, device=device)
        freq_y = _cis_from_positions(y_pos, self.inv_freq_y)
        freq_x = _cis_from_positions(x_pos, self.inv_freq_x)
        cis = torch.cat(
            (
                freq_y[:, None, :].expand(height, width, -1),
                freq_x[None, :, :].expand(height, width, -1),
            ),
            dim=-1,
        )
        return cis.reshape(height * width, -1)


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

    def forward(self, x: Tensor, cis: Tensor, *, num_registers: int = 0) -> Tensor:
        b, tokens, _ = x.shape
        h = self.norm1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(b, tokens, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(b, tokens, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(b, tokens, self.heads, self.head_dim).transpose(1, 2)
        if num_registers > 0:
            q = torch.cat(
                [q[:, :, :num_registers, :], apply_rope(q[:, :, num_registers:, :], cis)],
                dim=2,
            )
            k = torch.cat(
                [k[:, :, :num_registers, :], apply_rope(k[:, :, num_registers:, :], cis)],
                dim=2,
            )
        else:
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
        max_l5_grid_side: int,
        num_registers: int = 0,
    ) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by heads ({heads})")
        if num_registers < 0:
            raise ValueError(f"num_registers ({num_registers}) must be ≥ 0")
        head_dim = dim // heads
        if head_dim % 4 != 0:
            raise ValueError(
                f"head_dim ({head_dim}) must be divisible by 4 for 2D RoPE; "
                f"adjust dim or heads"
            )
        self.num_registers = num_registers
        if num_registers > 0:
            registers = torch.empty(1, num_registers, dim)
            nn.init.normal_(registers, std=0.02)
            self.registers = nn.Parameter(registers)
        else:
            self.registers = None
        self.in_proj = (
            nn.Conv2d(in_channels, dim, kernel_size=1)
            if in_channels != dim
            else nn.Identity()
        )
        self.rope = Rotary2D(head_dim, max_l5_grid_side=max_l5_grid_side)
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
        spatial = h.flatten(2).transpose(1, 2)
        if self.num_registers > 0:
            registers = self.registers.expand(b, -1, -1)
            tokens = torch.cat([registers, spatial], dim=1)
        else:
            tokens = spatial
        cis = self.rope.slice(height, width)
        for block in self.blocks:
            tokens = block(tokens, cis, num_registers=self.num_registers)
        tokens = self.norm(tokens)
        spatial = tokens[:, self.num_registers :, :]
        h = spatial.transpose(1, 2).view(b, -1, height, width)
        return x + self.out_proj(h)
