"""Learnable [state]/[question]/[options]/[candidate] segment marker embeddings."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerBase

from model.decision_model import ModernBERTDecisionModel

MARKER_NAMES: tuple[str, ...] = ("state", "question", "options", "candidate")
MARKER_TOKEN_STRINGS: tuple[str, ...] = (
    "[state]",
    "[question]",
    "[options]",
    "[candidate]",
)


def marker_token_ids(tokenizer: PreTrainedTokenizerBase) -> dict[str, int]:
    ids: dict[str, int] = {}
    for name, token in zip(MARKER_NAMES, MARKER_TOKEN_STRINGS, strict=True):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id < 0:
            raise RuntimeError(f"tokenizer missing segment marker token {token!r}")
        ids[name] = int(token_id)
    return ids


def ensure_marker_tokens(model: ModernBERTDecisionModel) -> None:
    tokenizer = model.tokenizer
    vocab = set(tokenizer.get_vocab().keys())
    missing = [t for t in MARKER_TOKEN_STRINGS if t not in vocab]
    if not missing:
        return
    added = tokenizer.add_special_tokens({"additional_special_tokens": missing})
    if added:
        model.encoder.resize_token_embeddings(len(tokenizer))
    model.base_vocab_size = len(tokenizer)


def _remove_embed_hook(model: ModernBERTDecisionModel) -> None:
    hook = getattr(model, "_embedding_forward_hook", None)
    if hook is not None:
        hook.remove()
        model._embedding_forward_hook = None


def _embedding_overlay_hook(model: ModernBERTDecisionModel):
    def hook(_module, inputs, output):
        input_ids = inputs[0]
        out = output
        for index, name in enumerate(MARKER_NAMES):
            if not model.marker_embedding_train[index]:
                continue
            param = model.marker_embedding_params[name]
            token_id = model.marker_token_ids[name]
            mask = input_ids == token_id
            if mask.any():
                out = out.clone()
                out[mask] = param.to(device=out.device, dtype=out.dtype)
        return out

    return hook


def init_marker_parameters(model: ModernBERTDecisionModel) -> None:
    ensure_marker_tokens(model)
    model.marker_token_ids = marker_token_ids(model.tokenizer)
    embed = model.encoder.get_input_embeddings()
    if embed is None:
        raise RuntimeError("encoder has no input embeddings")
    params = nn.ParameterDict()
    with torch.no_grad():
        for name in MARKER_NAMES:
            token_id = model.marker_token_ids[name]
            row = embed.weight[token_id].detach().clone()
            params[name] = nn.Parameter(row)
    model.marker_embedding_params = params


def apply_marker_embedding_trainability(
    model: ModernBERTDecisionModel,
    *,
    marker_embedding_train: Sequence[bool],
    marker_embedding_learning_rates: Sequence[float],
) -> None:
    if len(marker_embedding_train) != len(MARKER_NAMES):
        raise ValueError(
            f"marker_embedding_train length {len(marker_embedding_train)} "
            f"!= {len(MARKER_NAMES)}"
        )
    if len(marker_embedding_learning_rates) != len(MARKER_NAMES):
        raise ValueError(
            f"marker_embedding_learning_rates length {len(marker_embedding_learning_rates)} "
            f"!= {len(MARKER_NAMES)}"
        )
    model.marker_embedding_train = tuple(bool(x) for x in marker_embedding_train)
    model.marker_embedding_lrs = tuple(float(x) for x in marker_embedding_learning_rates)

    _remove_embed_hook(model)
    if not hasattr(model, "marker_embedding_params"):
        init_marker_parameters(model)

    for name in MARKER_NAMES:
        param = model.marker_embedding_params[name]
        param.requires_grad = model.marker_embedding_train[MARKER_NAMES.index(name)]

    embed = model.encoder.get_input_embeddings()
    if embed is not None:
        model._embedding_forward_hook = embed.register_forward_hook(
            _embedding_overlay_hook(model)
        )


def marker_embedding_param_groups(model: ModernBERTDecisionModel) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []
    for index, name in enumerate(MARKER_NAMES):
        if not model.marker_embedding_train[index]:
            continue
        param = model.marker_embedding_params[name]
        if param.requires_grad:
            groups.append({"params": [param], "lr": float(model.marker_embedding_lrs[index])})
    return groups


def marker_embedding_state(model: ModernBERTDecisionModel) -> dict[str, object] | None:
    if not any(model.marker_embedding_train):
        return None
    state: dict[str, object] = {}
    for index, name in enumerate(MARKER_NAMES):
        if not model.marker_embedding_train[index]:
            continue
        state[name] = {
            "token_id": model.marker_token_ids[name],
            "weight": model.marker_embedding_params[name].detach().cpu(),
        }
    return state or None


def apply_marker_embedding_state(model: ModernBERTDecisionModel, state: dict[str, object]) -> None:
    if not hasattr(model, "marker_token_ids"):
        init_marker_parameters(model)
    for name, entry in state.items():
        if name not in MARKER_NAMES:
            continue
        if not isinstance(entry, dict):
            raise ValueError(f"marker state for {name!r} must be a dict")
        row = entry.get("weight")
        if not isinstance(row, torch.Tensor):
            raise ValueError(f"marker {name!r} weight must be a tensor")
        param = model.marker_embedding_params[name]
        param.data.copy_(row.to(device=param.device, dtype=param.dtype))
        token_id = int(entry.get("token_id", model.marker_token_ids[name]))
        embed = model.encoder.get_input_embeddings()
        if embed is not None:
            embed.weight.data[token_id] = row.to(device=embed.weight.device, dtype=embed.weight.dtype)
