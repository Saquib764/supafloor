"""Parameter counts for pretrained vs trainable pieces."""

from __future__ import annotations

from dataclasses import dataclass

from model.decision_model import ModernBERTDecisionModel


@dataclass(frozen=True)
class ParameterBreakdown:
    encoder_trainable: int
    decision_stack: int
    score_head: int
    frozen: int
    trainable: int
    total: int


def parameter_breakdown(model: ModernBERTDecisionModel) -> ParameterBreakdown:
    stack = sum(p.numel() for p in model.decision_stack.parameters())
    head = sum(p.numel() for p in model.score_head.parameters())
    encoder_trainable = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return ParameterBreakdown(
        encoder_trainable=encoder_trainable,
        decision_stack=stack,
        score_head=head,
        frozen=frozen,
        trainable=trainable,
        total=total,
    )
