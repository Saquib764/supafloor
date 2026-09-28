"""Freeze pretrained encoder; optionally train last N layers, CLS row, and score head."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch

from model.decision_model import ModernBERTDecisionModel


def _zero_non_cls_embedding_grad(grad: torch.Tensor, cls_token_id: int) -> torch.Tensor:
    if grad is None:
        return grad
    out = grad.clone()
    mask = torch.ones_like(out, dtype=torch.bool)
    mask[cls_token_id] = False
    out[mask] = 0
    return out


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
    layers = model.encoder.layers
    start = len(layers) - n
    for slot in range(n):
        yield slot, start + slot


def encoder_finetune_param_groups(model: ModernBERTDecisionModel) -> list[dict[str, object]]:
    n = model.finetune_last_n_layers
    if n <= 0:
        return []
    train_mask = model.encoder_layer_train
    layer_lrs = model.encoder_finetune_lrs
    layers = model.encoder.layers
    groups: list[dict[str, object]] = []
    for slot, layer_idx in _finetune_slots(model):
        if not train_mask[slot]:
            continue
        params = [p for p in layers[layer_idx].parameters() if p.requires_grad]
        if params:
            groups.append({"params": params, "lr": float(layer_lrs[slot])})
    deepest = _deepest_trained_slot(train_mask)
    if deepest is not None:
        norm_params = [p for p in model.encoder.final_norm.parameters() if p.requires_grad]
        if norm_params:
            groups.append({"params": norm_params, "lr": float(layer_lrs[deepest])})
    return groups


def apply_encoder_trainability(
    model: ModernBERTDecisionModel,
    *,
    finetune_last_n_layers: int,
    encoder_layer_train: Sequence[bool],
    encoder_layer_learning_rates: Sequence[float],
    train_cls_embedding: bool,
) -> None:
    hook = getattr(model, "_embedding_grad_hook", None)
    if hook is not None:
        hook.remove()
        model._embedding_grad_hook = None

    n = max(0, int(finetune_last_n_layers))
    layers = model.encoder.layers
    if n > len(layers):
        raise ValueError(
            f"finetune_last_n_layers={n} exceeds encoder depth {len(layers)}"
        )
    train_mask = () if n == 0 else tuple(bool(x) for x in encoder_layer_train)
    finetune_lrs = () if n == 0 else tuple(float(x) for x in encoder_layer_learning_rates)

    model.finetune_last_n_layers = n
    model.encoder_layer_train = train_mask
    model.encoder_finetune_lrs = finetune_lrs
    model.train_cls_embedding = train_cls_embedding
    model.cls_token_id = None

    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.score_head.parameters():
        param.requires_grad = True

    for slot, layer_idx in _finetune_slots(model):
        if not train_mask[slot]:
            continue
        for param in layers[layer_idx].parameters():
            param.requires_grad = True
    if _deepest_trained_slot(train_mask) is not None:
        for param in model.encoder.final_norm.parameters():
            param.requires_grad = True

    if not train_cls_embedding:
        return

    cls_id = model.tokenizer.cls_token_id
    if cls_id is None:
        raise RuntimeError("tokenizer has no cls_token_id; cannot train CLS embedding")
    embeddings = model.encoder.get_input_embeddings()
    if embeddings is None:
        raise RuntimeError("encoder has no input embeddings")
    embeddings.weight.requires_grad = True
    model.cls_token_id = int(cls_id)

    def hook_fn(grad: torch.Tensor) -> torch.Tensor:
        return _zero_non_cls_embedding_grad(grad, model.cls_token_id)  # type: ignore[arg-type]

    model._embedding_grad_hook = embeddings.weight.register_hook(hook_fn)
