"""Floor-plan structure segmentation model."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from config import Config, L5TransformerSettings
from net.backbone import CnnBackbone
from net.fpn import YoloLikeFPN
from net.l5_transformer import L5Transformer
from net.mask_head import FPNMaskHead


@dataclass
class FloorplanFPNOutput:
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
        max_l5_grid_side=settings.max_l5_grid_side,
        num_registers=settings.num_registers,
    )


class FloorplanFPN(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 1,
        stem_channels: int = 32,
        backbone_channels: tuple[int, ...] = (48, 96, 128, 128),
        depth: int = 2,
        fpn_width: int = 128,
        p1_width: int = 32,
        out_channels: int = 1,
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
        self.fpn = YoloLikeFPN(
            self.backbone.out_channels,
            fpn_width,
            stem_channels=stem_channels,
            p1_width=p1_width,
            l5_transformer=l5_vit,
        )
        self.mask_head = FPNMaskHead(p1_width, fpn_width, out_channels=out_channels)

    def forward(self, x: Tensor) -> FloorplanFPNOutput:
        stem, features = self.backbone(x)
        pyramid = self.fpn(stem, features)
        logits = self.mask_head(pyramid["p1"], pyramid["p2"], pyramid["p3"])
        if logits.shape[-2:] != x.shape[-2:]:
            raise RuntimeError(
                f"mask logits {tuple(logits.shape)} != input spatial {tuple(x.shape[-2:])}"
            )
        return FloorplanFPNOutput(mask_logits=logits)


def build_model(cfg: Config) -> FloorplanFPN:
    return FloorplanFPN(
        in_channels=1,
        stem_channels=cfg.net.stem_channels,
        backbone_channels=cfg.net.backbone_channels,
        depth=cfg.net.depth,
        fpn_width=cfg.net.fpn_width,
        p1_width=cfg.net.p1_width,
        out_channels=cfg.net.out_channels,
        l5_transformer=cfg.net.l5_transformer,
    )
