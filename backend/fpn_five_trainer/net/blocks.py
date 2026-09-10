"""Shared conv blocks for the FPN network."""

from __future__ import annotations

from torch import Tensor, nn
from torch.nn import functional as F


class ConvBnSiLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        groups = min(8, out_channels)
        self.norm = nn.GroupNorm(groups, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return F.silu(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = ConvBnSiLU(channels, channels)
        self.conv2 = ConvBnSiLU(channels, channels)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.conv2(self.conv1(x))


class LargeKernelContext(nn.Module):
    """Depthwise 7×7 + pointwise 1×1, residual."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.pw = ConvBnSiLU(channels, channels, kernel_size=1, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pw(self.dw(x))
