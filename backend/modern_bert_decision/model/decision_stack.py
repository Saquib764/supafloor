"""Self-attention stack over per-option encoder hiddens (length K)."""

from __future__ import annotations

import torch
import torch.nn as nn


class OptionDecisionStack(nn.Module):
    """Transformer encoder over option token hiddens [B, K, H]."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        num_layers: int,
        dropout: float,
        pre_norm: bool = True,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("decision stack num_layers must be >= 1")
        if not pre_norm:
            raise ValueError("decision stack only supports pre_norm=true")
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_attention_heads,
            dim_feedforward=intermediate_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=pre_norm,
        )
        self.layers = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, hidden: torch.Tensor, option_mask: torch.Tensor) -> torch.Tensor:
        if hidden.dim() != 3:
            raise ValueError(f"decision stack expects [B, K, H], got {tuple(hidden.shape)}")
        if option_mask.shape != hidden.shape[:2]:
            raise ValueError(
                f"option_mask {tuple(option_mask.shape)} != batch/options {hidden.shape[:2]}"
            )
        pad = ~option_mask
        return self.layers(hidden, src_key_padding_mask=pad)
