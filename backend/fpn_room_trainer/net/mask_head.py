"""Fuse FPN levels and decode to full-resolution mask logits."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from net.blocks import ConvBnSiLU


class FPNMaskHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        *,
        out_channels: int = 1,
        hidden_channels: int = 64,
    ) -> None:
        super().__init__()
        fused = in_channels * 3
        self.fuse = nn.Sequential(
            ConvBnSiLU(fused, hidden_channels),
            ConvBnSiLU(hidden_channels, hidden_channels),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(self, p2: Tensor, p3: Tensor, p4: Tensor) -> Tensor:
        size = p2.shape[-2:]
        p3_up = F.interpolate(p3, size=size, mode="bilinear", align_corners=False)
        p4_up = F.interpolate(p4, size=size, mode="bilinear", align_corners=False)
        fused = torch.cat([p2, p3_up, p4_up], dim=1)
        logits = self.fuse(fused)
        return F.interpolate(logits, scale_factor=4, mode="bilinear", align_corners=False)
