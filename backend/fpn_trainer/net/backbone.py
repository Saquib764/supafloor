"""Multi-scale CNN encoder."""

from __future__ import annotations

from torch import Tensor, nn

from net.blocks import ConvBnSiLU, ResidualBlock


class DownStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, depth: int) -> None:
        super().__init__()
        self.down = ConvBnSiLU(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.blocks = nn.Sequential(*(ResidualBlock(out_channels) for _ in range(depth)))

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(self.down(x))


class CnnBackbone(nn.Module):
    """Return [c2, c3, c4, c5] at strides 4, 8, 16, 32."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        stem_channels: int = 32,
        stage_channels: tuple[int, ...] = (48, 96, 128, 128),
        depth: int = 2,
    ) -> None:
        super().__init__()
        if len(stage_channels) != 4:
            raise ValueError("stage_channels must have 4 entries for c2..c5")
        self.stem = ConvBnSiLU(in_channels, stem_channels, kernel_size=3, stride=2, padding=1)
        channels = [stem_channels, *stage_channels]
        self.stages = nn.ModuleList(
            DownStage(channels[index], channels[index + 1], depth=depth)
            for index in range(4)
        )
        self.out_channels = stage_channels

    def forward(self, x: Tensor) -> list[Tensor]:
        x = self.stem(x)
        features: list[Tensor] = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features
