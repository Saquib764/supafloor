"""Star-prompted room segmentation model."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from config import Config, L5TransformerSettings
from net.backbone import CnnBackbone
from net.fpn import YoloLikeFPN
from net.l5_transformer import L5Transformer
from net.mask_head import FPNMaskHead


@dataclass
class StarRoomFPNOutput:
    mask_logits: Tensor


def _build_l5_transformer(
    fpn_width: int,
    settings: L5TransformerSettings,
) -> L5Transformer:
    return L5Transformer(
        in_channels=fpn_width,
        dim=settings.dim,
        depth=settings.depth,
        heads=settings.heads,
        mlp_ratio=settings.mlp_ratio,
        level=settings.level,
        max_l5_grid_side=settings.max_l5_grid_side,
    )


class StarRoomFPN(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 3,
        stem_channels: int = 32,
        backbone_channels: tuple[int, ...] = (48, 96, 128, 128),
        depth: int = 2,
        fpn_width: int = 128,
        out_channels: int = 3,
        l5_transformer: L5TransformerSettings | None = None,
    ) -> None:
        super().__init__()
        self.backbone = CnnBackbone(
            in_channels=in_channels,
            stem_channels=stem_channels,
            stage_channels=backbone_channels,
            depth=depth,
        )
        l5_vit = _build_l5_transformer(fpn_width, l5_transformer) if l5_transformer else None
        self.fpn = YoloLikeFPN(self.backbone.out_channels, fpn_width, l5_transformer=l5_vit)
        self.mask_head = FPNMaskHead(fpn_width, out_channels=out_channels)

    def forward(self, x: Tensor) -> StarRoomFPNOutput:
        pyramid = self.fpn(self.backbone(x))
        logits = self.mask_head(pyramid["p2"], pyramid["p3"], pyramid["p4"])
        if logits.shape[-2:] != x.shape[-2:]:
            raise RuntimeError(
                f"mask logits {tuple(logits.shape)} != input spatial {tuple(x.shape[-2:])}"
            )
        return StarRoomFPNOutput(mask_logits=logits)


def build_model(cfg: Config) -> StarRoomFPN:
    return StarRoomFPN(
        stem_channels=cfg.net.stem_channels,
        backbone_channels=cfg.net.backbone_channels,
        depth=cfg.net.depth,
        fpn_width=cfg.net.fpn_width,
        out_channels=cfg.net.out_channels,
        l5_transformer=cfg.net.l5_transformer,
    )
