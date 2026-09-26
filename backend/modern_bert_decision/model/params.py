"""Parameter counts for pretrained vs newly added trainable pieces."""

from __future__ import annotations

from dataclasses import dataclass

from model.decision_model import ModernBERTDecisionModel

NUM_MARKER_TOKENS = 3


@dataclass(frozen=True)
class ParameterBreakdown:
    original_pretrained: int
    marker_embeddings: int
    score_head: int
    new_trainable: int
    frozen: int
    trainable: int
    total: int


def _param_bytes(n: int) -> int:
    return n * 4


def parameter_breakdown(model: ModernBERTDecisionModel) -> ParameterBreakdown:
    hidden = model.hidden_size
    marker = NUM_MARKER_TOKENS * hidden
    head = sum(p.numel() for p in model.score_head.parameters())
    encoder_total = sum(p.numel() for p in model.encoder.parameters())
    original = encoder_total - marker
    new_trainable = marker + head
    total = sum(p.numel() for p in model.parameters())
    if not model.train_pretrained_encoder:
        trainable = new_trainable
        frozen = total - trainable
    else:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = total - trainable
    return ParameterBreakdown(
        original_pretrained=original,
        marker_embeddings=marker,
        score_head=head,
        new_trainable=new_trainable,
        frozen=frozen,
        trainable=trainable,
        total=total,
    )

