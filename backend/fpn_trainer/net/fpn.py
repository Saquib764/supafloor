"""YOLO-style FPN neck: lateral + top-down fusion."""

from __future__ import annotations

from torch import Tensor, nn
from torch.nn import functional as F

from net.blocks import ConvBnSiLU
from net.l5_transformer import L5Transformer


class YoloLikeFPN(nn.Module):
    def __init__(
        self,
        in_channels: tuple[int, ...],
        out_channels: int,
        *,
        l5_transformer: L5Transformer | None = None,
    ) -> None:
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError("in_channels must have 4 entries for c2..c5")
        self.lateral = nn.ModuleList(
            ConvBnSiLU(channels, out_channels, kernel_size=1, padding=0)
            for channels in in_channels
        )
        self.l5_transformer = l5_transformer
        self.merge_td = nn.ModuleList(ConvBnSiLU(out_channels, out_channels) for _ in range(3))

    def forward(self, features: list[Tensor]) -> dict[str, Tensor]:
        c2, c3, c4, c5 = features
        l2, l3, l4, l5 = (lat(feat) for lat, feat in zip(self.lateral, features, strict=True))

        if self.l5_transformer is not None:
            l5 = self.l5_transformer(l5)

        p5 = l5
        p4 = self.merge_td[0](l4 + F.interpolate(p5, size=l4.shape[-2:], mode="nearest"))
        p3 = self.merge_td[1](l3 + F.interpolate(p4, size=l3.shape[-2:], mode="nearest"))
        p2 = self.merge_td[2](l2 + F.interpolate(p3, size=l2.shape[-2:], mode="nearest"))

        return {"p2": p2, "p3": p3, "p4": p4, "p5": p5}
