"""ModernBERT encoder with a shared linear scoring head over [OPTION] markers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from config import DecisionConfig, assert_hub_has_space, hub_from_pretrained_kwargs
from formatting import MarkerIds, marker_ids_from_tokenizer, max_sequence_length


@dataclass
class DecisionForwardOutput:
    logits: torch.Tensor
    probabilities: torch.Tensor
    selected_indices: torch.Tensor
    option_mask: torch.Tensor


class ModernBERTDecisionModel(nn.Module):
    def __init__(
        self,
        encoder: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        *,
        cfg: DecisionConfig,
        markers: MarkerIds,
        max_len: int,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.markers = markers
        self.max_len = max_len
        self.base_vocab_size = 0
        self.train_pretrained_encoder = False
        self._embedding_grad_hook = None
        hidden = int(encoder.config.hidden_size)
        self.score_head = nn.Linear(hidden, 1, bias=False)

    @property
    def hidden_size(self) -> int:
        return int(self.encoder.config.hidden_size)

    def save_pretrained(self, root: Path, *, at_end: bool = True) -> None:
        from checkpoint import save_checkpoint

        save_checkpoint(self, root, at_end=at_end)

    @staticmethod
    def load_pretrained(root: Path, cfg: DecisionConfig | None = None) -> ModernBERTDecisionModel:
        from checkpoint import load_checkpoint

        return load_checkpoint(root, cfg)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        option_positions: torch.Tensor,
        option_mask: torch.Tensor,
    ) -> DecisionForwardOutput:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        batch_size, k_max = option_positions.shape
        gather_index = option_positions.unsqueeze(-1).expand(batch_size, k_max, hidden.size(-1))
        option_hidden = torch.gather(hidden, dim=1, index=gather_index)
        logits = self.score_head(option_hidden).squeeze(-1)
        logits = logits.masked_fill(~option_mask, float("-inf"))

        probs = torch.softmax(logits.float(), dim=-1)
        probs = probs * option_mask.float()
        selected = torch.argmax(logits, dim=-1)
        return DecisionForwardOutput(
            logits=logits,
            probabilities=probs,
            selected_indices=selected,
            option_mask=option_mask,
        )


def _add_marker_tokens(tokenizer: PreTrainedTokenizerBase, cfg: DecisionConfig) -> None:
    tokens = [cfg.marker_tokens.state, cfg.marker_tokens.question, cfg.marker_tokens.option]
    tokenizer.add_special_tokens({"additional_special_tokens": tokens})


def _raise_load_failure(cfg: DecisionConfig, kind: str, exc: BaseException) -> None:
    hub = hub_from_pretrained_kwargs(cfg)["cache_dir"]
    msg = f"failed to load {kind} {cfg.pretrained_name!r} (hub cache: {hub})"
    err_text = str(exc).lower()
    if getattr(exc, "errno", None) == 28 or "no space left" in err_text:
        msg += (
            ". Disk full — free ~650 MiB on that volume or point hf_cache in config.yaml "
            "to a directory with space. Remove stale partial downloads under "
            f"{hub}/models--answerdotai--ModernBERT-base if a prior download failed."
        )
    raise RuntimeError(msg) from exc


def build_model(cfg: DecisionConfig) -> ModernBERTDecisionModel:
    hub_kw = hub_from_pretrained_kwargs(cfg)
    assert_hub_has_space(cfg)
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_name, use_fast=True, **hub_kw)
    except OSError as exc:
        _raise_load_failure(cfg, "tokenizer", exc)
    base_vocab_size = len(tokenizer)
    _add_marker_tokens(tokenizer, cfg)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.sep_token is not None:
            tokenizer.pad_token = tokenizer.sep_token
    try:
        encoder = AutoModel.from_pretrained(cfg.pretrained_name, **hub_kw)
    except OSError as exc:
        _raise_load_failure(cfg, "encoder", exc)
    except RuntimeError as exc:
        if "No space" in str(exc) or "disk" in str(exc).lower():
            _raise_load_failure(cfg, "encoder", exc)
        raise
    encoder.resize_token_embeddings(len(tokenizer))
    markers = marker_ids_from_tokenizer(tokenizer, cfg)
    max_len = max_sequence_length(tokenizer, encoder.config)
    model = ModernBERTDecisionModel(
        encoder,
        tokenizer,
        cfg=cfg,
        markers=markers,
        max_len=max_len,
    )
    from model.freeze import apply_encoder_trainability

    apply_encoder_trainability(
        model,
        base_vocab_size=base_vocab_size,
        train_pretrained_encoder=cfg.model.train_pretrained_encoder,
    )
    return model
