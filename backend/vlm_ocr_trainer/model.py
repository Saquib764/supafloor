"""~1M-parameter OCR VLM: CNN stem → ViT encoder → AR text decoder."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from vocab import CharVocab

GRID_SIZE = 16
VISION_DIM = 96
DECODER_DIM = 128
VISION_LAYERS = 5
DECODER_LAYERS = 3
VISION_HEADS = 4
DECODER_HEADS = 4
VISION_MLP = 192
DECODER_MLP = 256


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 2) -> None:
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


class Stride16Fusion(nn.Module):
    """Fuse stride-8 features into stride-16 tokens via learned downsample + patch merge."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.main = DepthwiseSeparableConv(in_channels, out_channels, stride=2)
        self.detail = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(in_channels * 4, out_channels, 1, bias=False),
        )
        self.norm = nn.GroupNorm(1, out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        fused = self.main(x) + self.detail(x)
        return self.act(self.norm(fused))


class CnnStem(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = VISION_DIM) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 24, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, 24),
            nn.SiLU(inplace=True),
            DepthwiseSeparableConv(24, 48),
            ResidualLocalBlock(48),
            DepthwiseSeparableConv(48, 72),
            ResidualLocalBlock(72),
            ResidualLocalBlock(72),
            Stride16Fusion(72, out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.stem(x)


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

    def apply(self, x: Tensor) -> Tensor:
        seq_len = x.shape[-2]
        return self._rotate(x, self.cos[..., :seq_len, :], self.sin[..., :seq_len, :])


class RoPE2D(nn.Module):
    """2D rotary position embeddings for an H×W token grid."""

    def __init__(self, head_dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError(f"head_dim ({head_dim}) must be divisible by 2 for 2D RoPE")
        axis_dim = head_dim // 2
        inv_freq = 1.0 / (theta ** (torch.arange(0, axis_dim, 2, dtype=torch.float32) / axis_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def apply(self, x: Tensor, grid_h: int, grid_w: int) -> Tensor:
        device = x.device
        dtype = self.inv_freq.dtype
        y_pos = torch.arange(grid_h, device=device, dtype=dtype)
        x_pos = torch.arange(grid_w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y_pos, x_pos, indexing="ij")
        flat_y = yy.reshape(-1)
        flat_x = xx.reshape(-1)
        cos_y = torch.cos(torch.outer(flat_y, self.inv_freq)).unsqueeze(0).unsqueeze(0)
        sin_y = torch.sin(torch.outer(flat_y, self.inv_freq)).unsqueeze(0).unsqueeze(0)
        cos_x = torch.cos(torch.outer(flat_x, self.inv_freq)).unsqueeze(0).unsqueeze(0)
        sin_x = torch.sin(torch.outer(flat_x, self.inv_freq)).unsqueeze(0).unsqueeze(0)
        half = x.shape[-1] // 2
        x_y, x_x = x.split(half, dim=-1)
        x_y = RoPE1D._rotate(x_y, cos_y, sin_y)
        x_x = RoPE1D._rotate(x_x, cos_x, sin_x)
        return torch.cat([x_y, x_x], dim=-1)


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        rope: RoPE2D | None = None,
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
        grid_h: int | None = None,
        grid_w: int | None = None,
    ) -> Tensor:
        batch, seq_len, _ = x.shape
        qkv = self.qkv(x).reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.rope is not None:
            if isinstance(self.rope, RoPE2D):
                if grid_h is None or grid_w is None:
                    raise ValueError("grid_h and grid_w are required for 2D RoPE")
                q = self.rope.apply(q, grid_h, grid_w)
                k = self.rope.apply(k, grid_h, grid_w)
            else:
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
        context_len = context.shape[1]
        q = self.q(x).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
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
    def __init__(self, dim: int, num_heads: int, mlp_dim: int, rope: RoPE2D) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, rope=rope)
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

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        x = x + self.self_attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), context)
        x = x + self.mlp(self.norm3(x))
        return x


class OcrVlm(nn.Module):
    def __init__(self, vocab: CharVocab) -> None:
        super().__init__()
        self.vocab = vocab
        self.stem = CnnStem()
        self.rope = RoPE2D(VISION_DIM // VISION_HEADS)
        self.vision_blocks = nn.ModuleList(
            VisionEncoderBlock(VISION_DIM, VISION_HEADS, VISION_MLP, self.rope)
            for _ in range(VISION_LAYERS)
        )
        self.vision_norm = nn.LayerNorm(VISION_DIM)
        self.vision_proj = nn.Linear(VISION_DIM, DECODER_DIM, bias=False)
        self.text_rope = RoPE1D(DECODER_DIM // DECODER_HEADS, max_seq_len=512)
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
        for block in self.vision_blocks:
            tokens = block(tokens, grid_h=grid_h, grid_w=grid_w)
        tokens = self.vision_norm(tokens)
        return self.vision_proj(tokens)

    def decode(self, visual_tokens: Tensor, token_ids: Tensor) -> Tensor:
        x = self.token_embedding(token_ids)
        for block in self.decoder_blocks:
            x = block(x, visual_tokens)
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
        max_length: int,
        bos_id: int | None = None,
        eos_id: int | None = None,
    ) -> Tensor:
        bos_id = self.vocab.bos_id if bos_id is None else bos_id
        eos_id = self.vocab.eos_id if eos_id is None else eos_id
        batch = images.shape[0]
        visual = self.encode_image(images)
        generated = torch.full((batch, 1), bos_id, dtype=torch.long, device=images.device)
        finished = torch.zeros(batch, dtype=torch.bool, device=images.device)
        for _ in range(max_length - 1):
            logits = self.decode(visual, generated)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            finished |= next_token.squeeze(1) == eos_id
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
