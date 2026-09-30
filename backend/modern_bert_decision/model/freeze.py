"""Freeze pretrained encoder; optionally train last N layers and score head."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch.nn as nn

from model.decision_model import ModernBERTDecisionModel


def _encoder_stack(model: ModernBERTDecisionModel) -> nn.Module:
    return model.encoder


def _deepest_trained_slot(train_mask: Sequence[bool]) -> int | None:
    for j in range(len(train_mask) - 1, -1, -1):
        if train_mask[j]:
            return j
    return None


def _finetune_slots(model: ModernBERTDecisionModel) -> Iterator[tuple[int, int]]:
    """Yield (slot_index, encoder_layer_index) for the finetune window."""
    n = model.finetune_last_n_layers
    if n <= 0:
        return
    layers = _encoder_stack(model).layers
    start = len(layers) - n
    for slot in range(n):
        yield slot, start + slot


def encoder_finetune_param_groups(model: ModernBERTDecisionModel) -> list[dict[str, object]]:
    n = model.finetune_last_n_layers
    if n <= 0:
        return []
    train_mask = model.encoder_layer_train
    layer_lrs = model.encoder_finetune_lrs
    stack = _encoder_stack(model)
    layers = stack.layers
    groups: list[dict[str, object]] = []
    for slot, layer_idx in _finetune_slots(model):
        if not train_mask[slot]:
            continue
        params = [p for p in layers[layer_idx].parameters() if p.requires_grad]
        if params:
            groups.append({"params": params, "lr": float(layer_lrs[slot])})
    deepest = _deepest_trained_slot(train_mask)
    if deepest is not None:
        norm_params = [p for p in stack.final_norm.parameters() if p.requires_grad]
        if norm_params:
            groups.append({"params": norm_params, "lr": float(layer_lrs[deepest])})
    return groups


def apply_encoder_trainability(
    model: ModernBERTDecisionModel,
    *,
    finetune_last_n_layers: int,
    encoder_layer_train: Sequence[bool],
    encoder_layer_learning_rates: Sequence[float],
) -> None:
    stack = _encoder_stack(model)
    layers = stack.layers
    n = max(0, int(finetune_last_n_layers))
    if n > len(layers):
        raise ValueError(
            f"finetune_last_n_layers={n} exceeds encoder depth {len(layers)}"
        )
    train_mask = () if n == 0 else tuple(bool(x) for x in encoder_layer_train)
    finetune_lrs = () if n == 0 else tuple(float(x) for x in encoder_layer_learning_rates)

    model.finetune_last_n_layers = n
    model.encoder_layer_train = train_mask
    model.encoder_finetune_lrs = finetune_lrs

    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.decision_stack.parameters():
        param.requires_grad = True
    for param in model.score_head.parameters():
        param.requires_grad = True

    for slot, layer_idx in _finetune_slots(model):
        if not train_mask[slot]:
            continue
        for param in layers[layer_idx].parameters():
            param.requires_grad = True
    if _deepest_trained_slot(train_mask) is not None:
        for param in stack.final_norm.parameters():
            param.requires_grad = True
