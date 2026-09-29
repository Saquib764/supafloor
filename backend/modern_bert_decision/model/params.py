"""Parameter counts for pretrained vs trainable pieces."""

from __future__ import annotations

from dataclasses import dataclass

from model.decision_model import ModernBERTDecisionModel
from model.lora import lora_trainable_parameter_count


@dataclass(frozen=True)
class ParameterBreakdown:
    encoder_trainable: int
    lora: int
    score_head: int
    frozen: int
    trainable: int
    total: int


def parameter_breakdown(model: ModernBERTDecisionModel) -> ParameterBreakdown:
    head = sum(p.numel() for p in model.score_head.parameters())
    lora = lora_trainable_parameter_count(model.encoder) if model.lora_enabled else 0
    encoder_trainable = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    encoder_blocks = max(0, encoder_trainable - lora)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return ParameterBreakdown(
        encoder_trainable=encoder_blocks,
        lora=lora,
        score_head=head,
        frozen=frozen,
        trainable=trainable,
        total=total,
    )
