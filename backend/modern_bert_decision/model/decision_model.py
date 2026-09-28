"""ModernBERT encoder with a CLS linear score head (one forward pass per candidate)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from config import DecisionConfig, assert_hub_has_space, hub_from_pretrained_kwargs


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
        max_len: int,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.base_vocab_size = len(tokenizer)
        self.finetune_last_n_layers = 0
        self.encoder_layer_train: tuple[bool, ...] = ()
        self.encoder_finetune_lrs: tuple[float, ...] = ()
        self.train_cls_embedding = False
        self.marker_token_ids: dict[str, int] = {}
        self.marker_embedding_params = nn.ParameterDict()
        self.marker_embedding_train: tuple[bool, ...] = ()
        self.marker_embedding_lrs: tuple[float, ...] = ()
        self._embedding_forward_hook = None
        self.cls_token_id: int | None = None
        self._embedding_grad_hook = None
        hidden = int(encoder.config.hidden_size)
        self.score_head_dropout = nn.Dropout(p=float(cfg.model.score_head_dropout))
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

    def _cls_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = outputs.last_hidden_state[:, 0, :]
        cls_hidden = self.score_head_dropout(cls_hidden)
        return self.score_head(cls_hidden).squeeze(-1)

    def score_gold(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Training: one gold-verification sequence per example → logits [B]."""
        if input_ids.dim() != 2:
            raise ValueError(f"score_gold expects [B, L], got {tuple(input_ids.shape)}")
        return self._cls_logits(input_ids, attention_mask)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        option_mask: torch.Tensor,
    ) -> DecisionForwardOutput:
        """Inference/eval: one logit per option; softmax over valid options for the decision."""
        if input_ids.dim() != 3:
            raise ValueError(f"forward expects [B, K, L], got shape {tuple(input_ids.shape)}")
        batch_size, k_max, seq_len = input_ids.shape
        flat_ids = input_ids.reshape(batch_size * k_max, seq_len)
        flat_mask = attention_mask.reshape(batch_size * k_max, seq_len)
        flat_logits = self._cls_logits(flat_ids, flat_mask)
        logits = flat_logits.reshape(batch_size, k_max)

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
    from formatting import max_sequence_length

    max_len = max_sequence_length(tokenizer, encoder.config)
    model = ModernBERTDecisionModel(
        encoder,
        tokenizer,
        cfg=cfg,
        max_len=max_len,
    )
    from model.freeze import apply_encoder_trainability
    from model.segment_markers import apply_marker_embedding_trainability, init_marker_parameters

    init_marker_parameters(model)
    apply_encoder_trainability(
        model,
        finetune_last_n_layers=cfg.model.finetune_last_n_layers,
        encoder_layer_train=cfg.train.encoder_layer_train,
        encoder_layer_learning_rates=cfg.train.encoder_layer_learning_rates,
        train_cls_embedding=cfg.train.train_cls_embedding,
    )
    apply_marker_embedding_trainability(
        model,
        marker_embedding_train=cfg.train.marker_embedding_train,
        marker_embedding_learning_rates=cfg.train.marker_embedding_learning_rates,
    )
    return model
