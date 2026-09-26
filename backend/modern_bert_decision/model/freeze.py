"""Freeze pretrained encoder weights; keep marker embeddings and score head trainable."""

from __future__ import annotations

from typing import Any

import torch

from model.decision_model import ModernBERTDecisionModel


def _zero_pretrained_embedding_grad(grad: torch.Tensor, base_vocab_size: int) -> torch.Tensor:
    if grad is None:
        return grad
    out = grad.clone()
    out[:base_vocab_size] = 0
    return out


def apply_encoder_trainability(
    model: ModernBERTDecisionModel,
    *,
    base_vocab_size: int,
    train_pretrained_encoder: bool,
) -> None:
    hook = getattr(model, "_embedding_grad_hook", None)
    if hook is not None:
        hook.remove()
        model._embedding_grad_hook = None

    model.base_vocab_size = base_vocab_size
    model.train_pretrained_encoder = train_pretrained_encoder

    if train_pretrained_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = True
        for param in model.score_head.parameters():
            param.requires_grad = True
        return

    for param in model.encoder.parameters():
        param.requires_grad = False

    embeddings = model.encoder.get_input_embeddings()
    if embeddings is None:
        raise RuntimeError("encoder has no input embeddings")
    weight = embeddings.weight
    weight.requires_grad = True

    def hook_fn(grad: torch.Tensor) -> torch.Tensor:
        return _zero_pretrained_embedding_grad(grad, base_vocab_size)

    model._embedding_grad_hook: Any = weight.register_hook(hook_fn)

    for param in model.score_head.parameters():
        param.requires_grad = True
