"""ModernBERT encoder + option decision stack + MLP score head."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from config import DecisionConfig, assert_hub_has_space, hub_from_pretrained_kwargs
from model.decision_stack import OptionDecisionStack
from model.score_head import OptionScoreHead


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
        enc_hidden = int(encoder.config.hidden_size)
        dl = cfg.model.decision_layers
        sh = cfg.model.score_head
        if dl.hidden_size != enc_hidden:
            raise ValueError(
                f"model.decision_layers.hidden_size {dl.hidden_size} must match "
                f"encoder hidden_size {enc_hidden}"
            )
        if sh.hidden_size != dl.hidden_size:
            raise ValueError("model.score_head.hidden_size must match model.decision_layers.hidden_size")
        self.option_dropout = nn.Dropout(p=float(sh.dropout))
        self.decision_stack = OptionDecisionStack(
            hidden_size=dl.hidden_size,
            num_attention_heads=dl.num_attention_heads,
            intermediate_size=dl.intermediate_size,
            num_layers=dl.count,
            dropout=dl.dropout,
            pre_norm=dl.pre_norm,
        )
        self.score_head = OptionScoreHead(sh.hidden_size)

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
        score_index: torch.Tensor,
        option_mask: torch.Tensor,
    ) -> DecisionForwardOutput:
        """Gather score-token hiddens, run decision stack over options, then MLP score head."""
        if input_ids.dim() != 2:
            raise ValueError(f"forward expects [B, L], got shape {tuple(input_ids.shape)}")
        if score_index.shape != option_mask.shape:
            raise ValueError(
                f"score_index shape {tuple(score_index.shape)} != option_mask {tuple(option_mask.shape)}"
            )
        batch_size, k_max = score_index.shape
        if option_mask.shape[0] != batch_size:
            raise ValueError("batch size mismatch between input_ids and option_mask")

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        batch_idx = torch.arange(batch_size, device=hidden.device).unsqueeze(1).expand(-1, k_max)
        idx = score_index.clamp(min=0).long()
        gathered = hidden[batch_idx, idx]
        gathered = self.option_dropout(gathered)
        refined = self.decision_stack(gathered, option_mask)
        logits = self.score_head(refined)

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
            f"{hub}/models--{cfg.pretrained_name.replace('/', '--')} if a prior download failed."
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
    from formatting import resolve_sequence_length

    max_len = resolve_sequence_length(tokenizer, encoder.config, cap=cfg.model.max_len)
    model = ModernBERTDecisionModel(
        encoder,
        tokenizer,
        cfg=cfg,
        max_len=max_len,
    )
    configure_for_training(model, cfg)
    return model


def configure_for_training(model: ModernBERTDecisionModel, cfg: DecisionConfig) -> None:
    from model.freeze import apply_encoder_trainability

    apply_encoder_trainability(
        model,
        finetune_last_n_layers=cfg.model.finetune_last_n_layers,
        encoder_layer_train=cfg.train.encoder_layer_train,
        encoder_layer_learning_rates=cfg.train.encoder_layer_learning_rates,
    )
