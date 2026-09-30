"""Per-option scorer: LayerNorm → Linear → GELU → Linear → logit."""

from __future__ import annotations

import torch
import torch.nn as nn


class OptionScoreHead(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        last = self.net[-1]
        assert isinstance(last, nn.Linear)
        nn.init.normal_(last.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(last.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.dim() != 3:
            raise ValueError(f"score head expects [B, K, H], got {tuple(hidden.shape)}")
        return self.net(hidden).squeeze(-1)
