"""~1M-parameter OCR VLM: CNN stem → ViT encoder → AR text decoder."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

AxialAxis = Literal["horizontal", "vertical"]

from vocab import CharVocab

STEM_STRIDE_W = 4
STEM_STRIDE_H = 8
VISION_DIM = 96
DECODER_DIM = 112
VISION_LAYERS = 8
DECODER_LAYERS = 2
VISION_HEADS = 4
DECODER_HEADS = 4
VISION_MLP = 192
DECODER_MLP = 256
DECODER_MAX_SEQ_LEN = 2400


def validate_image_size(width: int, height: int) -> None:
    if width < STEM_STRIDE_W or height < STEM_STRIDE_H:
        raise ValueError(
            f"image size {width}x{height} must be at least "
            f"{STEM_STRIDE_W}x{STEM_STRIDE_H}",
        )
    if width % STEM_STRIDE_W or height % STEM_STRIDE_H:
        raise ValueError(
            f"image size {width}x{height} must be divisible by "
            f"{STEM_STRIDE_W} (width) and {STEM_STRIDE_H} (height)",
        )


def visual_grid(width: int, height: int) -> tuple[int, int]:
    """Return visual token grid (width, height) after the CNN stem."""
    return width // STEM_STRIDE_W, height // STEM_STRIDE_H


def visual_token_count(width: int, height: int) -> int:
    grid_w, grid_h = visual_grid(width, height)
    return grid_w * grid_h


def layer_order() -> tuple[str, ...]:
    layers = ["stem"]
    layers.extend(f"vision_blocks.{index}" for index in range(VISION_LAYERS))
    layers.extend(["vision_norm", "vision_proj", "token_embedding"])
    layers.extend(f"decoder_blocks.{index}" for index in range(DECODER_LAYERS))
    layers.append("decoder_norm")
    return tuple(layers)


def _match_layer(name: str, layer_order: tuple[str, ...]) -> str:
    if name.startswith("lm_head."):
        return "token_embedding"
    matched = next(
        (key for key in layer_order if name == key or name.startswith(f"{key}.")),
        None,
    )
    if matched is None:
        raise RuntimeError(f"parameter not in known layers: {name}")
    return matched


def layer_param_counts(model: OcrVlm) -> dict[str, tuple[int, int]]:
    order = layer_order()
    counts = {name: (0, 0) for name in order}
    seen: set[int] = set()
    for name, param in model.named_parameters():
        layer = _match_layer(name, order)
        storage = param.untyped_storage().data_ptr()
        if storage in seen:
            continue
        seen.add(storage)
        count, size = counts[layer]
        n = param.numel()
        counts[layer] = (count + n, size + n * param.element_size())
    return counts


def print_architecture_summary(
    *,
    canvas_sizes: tuple[tuple[int, int], ...],
    example_width: int,
    example_height: int,
    vocab_size: int,
) -> None:
    s0w, s0h = example_width, example_height
    s1w, s1h = example_width // 2, example_height // 2
    s2w, s2h = example_width // 4, example_height // 4
    grid_w, grid_h = visual_grid(example_width, example_height)
    sizes = ", ".join(f"{w}×{h}" for w, h in canvas_sizes)

    print("\n=== Architecture ===", flush=True)
    print("Task:     grayscale text image → character transcription (autoregressive)", flush=True)
    print(f"Input:    (1, H, W)   gray, normalized [-1, 1]   sizes: {sizes}", flush=True)
    print(
        f"Stem:     W÷{STEM_STRIDE_W}, H÷{STEM_STRIDE_H} "
        f"(e.g. {example_width}×{example_height} → {grid_w}×{grid_h} tokens)",
        flush=True,
    )
    print(f"Output:   (seq, {vocab_size})   char logits → BOS…text…EOS", flush=True)
    print(flush=True)
    print("OcrVlm", flush=True)
    print("├─ CnnStem", flush=True)
    print(f"│  ├─ conv 3×3 stride 1     1→16 ch   @ {s0w}×{s0h}", flush=True)
    print(f"│  ├─ conv 3×3 stride 2     16→24 ch  @ {s1w}×{s1h}", flush=True)
    print(f"│  ├─ dwsep stride 2        24→48 ch  @ {s2w}×{s2h}", flush=True)
    print(f"│  ├─ local block           48→48 ch  @ {s2w}×{s2h}  ×3", flush=True)
    print(
        f"│  ├─ dwsep stride (2,1)    48→72 ch  @ {grid_w}×{grid_h}  vertical only",
        flush=True,
    )
    print(f"│  ├─ local block           72→72 ch  @ {grid_w}×{grid_h}  ×2", flush=True)
    print(f"│  └─ conv 1×1              72→96 ch  @ {grid_w}×{grid_h}", flush=True)
    print(
        f"├─ VisionEncoder ({VISION_LAYERS} layers, d={VISION_DIM}, "
        f"heads={VISION_HEADS}, mlp={VISION_MLP}, axial H/V, 2D RoPE + fixed 2D pos)",
        flush=True,
    )
    for index in range(VISION_LAYERS):
        axis = "H" if index % 2 == 0 else "V"
        branch = "│  └─" if index == VISION_LAYERS - 1 else "│  ├─"
        print(f"{branch} block {index}   {axis} self-attn", flush=True)
    print(f"├─ vision_proj            {VISION_DIM}→{DECODER_DIM}", flush=True)
    print(
        f"├─ TextDecoder ({DECODER_LAYERS} layers, d={DECODER_DIM}, "
        f"heads={DECODER_HEADS}, mlp={DECODER_MLP}, 1D RoPE)",
        flush=True,
    )
    print(f"│  ├─ token_embedding     {vocab_size}×{DECODER_DIM} (weight-tied lm_head)", flush=True)
    for index in range(DECODER_LAYERS):
        branch = "│  └─" if index == DECODER_LAYERS - 1 else "│  ├─"
        print(
            f"{branch} block {index}   causal self-attn + cross-attn → visual tokens",
            flush=True,
        )
    print(f"└─ lm_head                {DECODER_DIM}→{vocab_size} (tied)", flush=True)


def print_vision_encode_summary(
    *,
    canvas_sizes: tuple[tuple[int, int], ...],
    example_width: int,
    example_height: int,
    max_text_len: int,
    max_seq_len: int,
) -> None:
    sizes = ", ".join(f"{w}×{h}" for w, h in canvas_sizes)
    grid_w, grid_h = visual_grid(example_width, example_height)
    visual_tokens = visual_token_count(example_width, example_height)

    print("\n=== Input ===", flush=True)
    print("Source:   synthetic grayscale PIL image (random charset text)", flush=True)
    print(f"Canvas:   one of [{sizes}] px (width×height), chosen per batch", flush=True)
    print(f"Text:     random vocabulary chars, up to {max_text_len} per sample", flush=True)
    print(flush=True)
    print("Preprocess pipeline:", flush=True)
    print("  1. PIL grayscale (L) → float32 numpy", flush=True)
    print("  2. Normalize:  x = pixel / 127.5 − 1   (range [-1, 1])", flush=True)
    print("  3. Tensor shape:  [1, H, W]", flush=True)
    print("  4. Batch: one canvas size per batch → torch.stack (no image padding)", flush=True)
    print("  5. Labels: pad shorter token sequences with <PAD> (ignored in loss)", flush=True)
    print(flush=True)
    print("Vision encode (fixed for the whole decode loop):", flush=True)
    print("  image [1, 1, H, W]", flush=True)
    print(f"    → CNN stem W÷{STEM_STRIDE_W}, H÷{STEM_STRIDE_H}", flush=True)
    print("    → ViT encoder + H/V axial attention, 2D RoPE, fixed 2D pos", flush=True)
    print(f"    → linear {VISION_DIM}→{DECODER_DIM}", flush=True)
    print(
        f"    → visual tokens [1, W/{STEM_STRIDE_W}·H/{STEM_STRIDE_H}, {DECODER_DIM}]"
        f"   (e.g. {example_width}×{example_height} → {grid_w}×{grid_h} = {visual_tokens} tokens)",
        flush=True,
    )
    print(flush=True)
    print("Training label (teacher forcing):", flush=True)
    print('  text  "Room 203\\nWidth: 4.5m"', flush=True)
    print(f"  ids   [<BOS>, R, o, o, m, …, <EOS>]   (max {max_seq_len} tokens)", flush=True)
    print("  loss  cross-entropy on next-char prediction; <PAD> ignored", flush=True)


class DepthwiseSeparableConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int | tuple[int, int] = 2,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.norm = nn.GroupNorm(1, out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.act(self.norm(x))


class ResidualLocalBlock(nn.Module):
    """Stride-preserving depthwise local mixing with a residual connection."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = DepthwiseSeparableConv(channels, channels, stride=1)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.conv(x)


class CnnStem(nn.Module):
    """Anisotropic CNN stem: W÷4, H÷8. 256×256 → 64×32 visual grid (2048 tokens)."""

    def __init__(self, in_channels: int = 1, out_channels: int = VISION_DIM) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=1, padding=1, bias=False),
            nn.GroupNorm(1, 16),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 24, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, 24),
            nn.SiLU(inplace=True),
            DepthwiseSeparableConv(24, 48),
            ResidualLocalBlock(48),
            ResidualLocalBlock(48),
            ResidualLocalBlock(48),
            DepthwiseSeparableConv(48, 72, stride=(2, 1)),
            ResidualLocalBlock(72),
            ResidualLocalBlock(72),
            nn.Conv2d(72, out_channels, 1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.stem(x)


class RoPE2D(nn.Module):
    """2D rotary embeddings; axial blocks rotate the y or x half only."""

    def __init__(self, head_dim: int, max_len: int = 512, theta: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError(f"head_dim ({head_dim}) must be divisible by 2")
        axis_dim = head_dim // 2
        inv_freq = 1.0 / (theta ** (torch.arange(0, axis_dim, 2, dtype=torch.float32) / axis_dim))
        positions = torch.arange(max_len, dtype=torch.float32)
        angles = torch.outer(positions, inv_freq)
        cos = torch.cos(angles).unsqueeze(0).unsqueeze(0)
        sin = torch.sin(angles).unsqueeze(0).unsqueeze(0)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _rotate_half(self, x: Tensor, seq_len: int) -> Tensor:
        half = x.shape[-1] // 2
        x_rot = x[..., half:]
        x_rot = RoPE1D._rotate(x_rot, self.cos[..., :seq_len, :], self.sin[..., :seq_len, :])
        return torch.cat([x[..., :half], x_rot], dim=-1)

    def _rotate_full(self, x: Tensor, seq_len: int) -> Tensor:
        return RoPE1D._rotate(x, self.cos[..., :seq_len, :], self.sin[..., :seq_len, :])

    def apply_horizontal(self, x: Tensor) -> Tensor:
        return self._rotate_half(x, x.shape[-2])

    def apply_vertical(self, x: Tensor) -> Tensor:
        half = x.shape[-1] // 2
        x_y, x_x = x.split(half, dim=-1)
        x_y = self._rotate_full(x_y, x.shape[-2])
        return torch.cat([x_y, x_x], dim=-1)


class FixedPosEmbed2D(nn.Module):
    """Fixed sinusoidal 2D position features added to vision tokens."""

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, grid_h: int, grid_w: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        dim_y = self.dim // 2
        dim_x = self.dim - dim_y
        ys = torch.arange(grid_h, device=device, dtype=dtype)
        xs = torch.arange(grid_w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")

        def axis_embed(pos: Tensor, axis_dim: int) -> Tensor:
            inv_freq = 1.0 / (
                self.theta ** (torch.arange(0, axis_dim, 2, device=device, dtype=dtype) / axis_dim)
            )
            angles = pos.reshape(-1, 1) * inv_freq
            return torch.cat([angles.sin(), angles.cos()], dim=-1)

        embed = torch.cat([axis_embed(yy, dim_y), axis_embed(xx, dim_x)], dim=-1)
        return embed.unsqueeze(0)


class RoPE1D(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 512, theta: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError(f"head_dim ({head_dim}) must be divisible by 2")
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        angles = torch.outer(positions, inv_freq)
        cos = torch.cos(angles).unsqueeze(0).unsqueeze(0)
        sin = torch.sin(angles).unsqueeze(0).unsqueeze(0)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @staticmethod
    def _rotate(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    def apply(self, x: Tensor, *, offset: int = 0) -> Tensor:
        seq_len = x.shape[-2]
        return self._rotate(
            x,
            self.cos[..., offset : offset + seq_len, :],
            self.sin[..., offset : offset + seq_len, :],
        )


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        rope: RoPE1D | None = None,
        causal: bool = False,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.causal = causal
        self.rope = rope
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x: Tensor,
        *,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        batch, seq_len, _ = x.shape
        qkv = self.qkv(x).reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.rope is not None:
            q = self.rope.apply(q)
            k = self.rope.apply(k)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask[:, None, None, :], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        context = torch.matmul(weights, v)
        context = context.transpose(1, 2).reshape(batch, seq_len, -1)
        return self.out(context)


class MultiHeadAxialSelfAttention(nn.Module):
    """Bidirectional self-attention along one grid axis (row or column)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        axis: AxialAxis,
        rope: RoPE2D,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.axis = axis
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.rope = rope
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: Tensor, *, grid_h: int, grid_w: int) -> Tensor:
        batch, seq_len, dim = x.shape
        if seq_len != grid_h * grid_w:
            raise ValueError(f"seq_len ({seq_len}) != grid_h*grid_w ({grid_h * grid_w})")

        x = x.view(batch, grid_h, grid_w, dim)
        if self.axis == "horizontal":
            x = x.reshape(batch * grid_h, grid_w, dim)
        else:
            x = x.permute(0, 2, 1, 3).reshape(batch * grid_w, grid_h, dim)

        sub_batch, axis_len, _ = x.shape
        qkv = self.qkv(x).reshape(sub_batch, axis_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.axis == "horizontal":
            q = self.rope.apply_horizontal(q)
            k = self.rope.apply_horizontal(k)
        else:
            q = self.rope.apply_vertical(q)
            k = self.rope.apply_vertical(k)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        weights = F.softmax(scores, dim=-1)
        context = torch.matmul(weights, v)
        context = context.transpose(1, 2).reshape(sub_batch, axis_len, dim)
        x = self.out(context)

        if self.axis == "horizontal":
            x = x.view(batch, grid_h, grid_w, dim)
        else:
            x = x.view(batch, grid_w, grid_h, dim).permute(0, 2, 1, 3)
        return x.reshape(batch, grid_h * grid_w, dim)


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, dim: int, context_dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(context_dim, dim * 2, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        batch, seq_len, _ = x.shape
        q = self.q(x).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        context_len = context.shape[1]
        kv = self.kv(context).reshape(batch, context_len, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        weights = F.softmax(scores, dim=-1)
        context_out = torch.matmul(weights, v)
        context_out = context_out.transpose(1, 2).reshape(batch, seq_len, -1)
        return self.out(context_out)


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class VisionEncoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_dim: int,
        *,
        axis: AxialAxis,
        rope: RoPE2D,
    ) -> None:
        super().__init__()
        self.axis = axis
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAxialSelfAttention(dim, num_heads, axis=axis, rope=rope)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_dim)

    def forward(self, x: Tensor, *, grid_h: int, grid_w: int) -> Tensor:
        x = x + self.attn(self.norm1(x), grid_h=grid_h, grid_w=grid_w)
        x = x + self.mlp(self.norm2(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        context_dim: int,
        num_heads: int,
        mlp_dim: int,
        rope: RoPE1D,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = MultiHeadSelfAttention(dim, num_heads, rope=rope, causal=True)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = MultiHeadCrossAttention(dim, context_dim, num_heads)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_dim)

    def forward(
        self,
        x: Tensor,
        context: Tensor,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        x = x + self.self_attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.cross_attn(self.norm2(x), context)
        x = x + self.mlp(self.norm3(x))
        return x


class OcrVlm(nn.Module):
    def __init__(self, vocab: CharVocab) -> None:
        super().__init__()
        self.vocab = vocab
        self.stem = CnnStem()
        self.fixed_pos_embed = FixedPosEmbed2D(VISION_DIM)
        self.vision_rope = RoPE2D(VISION_DIM // VISION_HEADS, max_len=512)
        self.vision_blocks = nn.ModuleList(
            VisionEncoderBlock(
                VISION_DIM,
                VISION_HEADS,
                VISION_MLP,
                axis="horizontal" if index % 2 == 0 else "vertical",
                rope=self.vision_rope,
            )
            for index in range(VISION_LAYERS)
        )
        self.vision_norm = nn.LayerNorm(VISION_DIM)
        self.vision_proj = nn.Linear(VISION_DIM, DECODER_DIM, bias=False)
        self.text_rope = RoPE1D(
            DECODER_DIM // DECODER_HEADS,
            max_seq_len=DECODER_MAX_SEQ_LEN,
        )
        self.token_embedding = nn.Embedding(vocab.size, DECODER_DIM)
        self.decoder_blocks = nn.ModuleList(
            DecoderBlock(DECODER_DIM, DECODER_DIM, DECODER_HEADS, DECODER_MLP, self.text_rope)
            for _ in range(DECODER_LAYERS)
        )
        self.decoder_norm = nn.LayerNorm(DECODER_DIM)
        self.lm_head = nn.Linear(DECODER_DIM, vocab.size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def encode_image(self, images: Tensor) -> Tensor:
        features = self.stem(images)
        _batch, _channels, grid_h, grid_w = features.shape
        tokens = features.flatten(2).transpose(1, 2)
        tokens = tokens + self.fixed_pos_embed(
            grid_h,
            grid_w,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        for block in self.vision_blocks:
            tokens = block(tokens, grid_h=grid_h, grid_w=grid_w)
        tokens = self.vision_norm(tokens)
        return self.vision_proj(tokens)

    def decode(self, visual_tokens: Tensor, token_ids: Tensor) -> Tensor:
        x = self.token_embedding(token_ids)
        attn_mask = token_ids != self.vocab.pad_id
        for block in self.decoder_blocks:
            x = block(x, visual_tokens, attn_mask=attn_mask)
        x = self.decoder_norm(x)
        return self.lm_head(x)

    def forward(self, images: Tensor, token_ids: Tensor) -> Tensor:
        visual = self.encode_image(images)
        return self.decode(visual, token_ids[:, :-1])

    @torch.no_grad()
    def generate(
        self,
        images: Tensor,
        *,
        max_length: int | Sequence[int],
        bos_id: int | None = None,
        eos_id: int | None = None,
    ) -> Tensor:
        bos_id = self.vocab.bos_id if bos_id is None else bos_id
        eos_id = self.vocab.eos_id if eos_id is None else eos_id
        batch = images.shape[0]
        if isinstance(max_length, int):
            length_limits = torch.full(
                (batch,),
                max_length,
                dtype=torch.long,
                device=images.device,
            )
        else:
            if len(max_length) != batch:
                raise ValueError(
                    f"max_length length ({len(max_length)}) != batch size ({batch})",
                )
            length_limits = torch.tensor(max_length, dtype=torch.long, device=images.device)
        visual = self.encode_image(images)
        generated = torch.full((batch, 1), bos_id, dtype=torch.long, device=images.device)
        finished = torch.zeros(batch, dtype=torch.bool, device=images.device)
        for _ in range(int(length_limits.max().item()) - 1):
            logits = self.decode(visual, generated)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            next_token = next_token.masked_fill(finished.unsqueeze(1), eos_id)
            generated = torch.cat([generated, next_token], dim=1)
            finished |= next_token.squeeze(1) == eos_id
            finished |= generated.shape[1] >= length_limits
            if finished.all():
                break
        return generated


def ocr_loss(logits: Tensor, targets: Tensor, *, pad_id: int) -> Tensor:
    """Teacher-forced cross entropy; ignore `<PAD>` positions."""
    vocab_size = logits.shape[-1]
    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
        ignore_index=pad_id,
    )
    return loss


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())
