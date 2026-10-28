"""Qwen3 blocks + learnable probe tokens + multi-layer linear head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from config import Config, hub_from_pretrained_kwargs
from data import prepare_tokenizer
from labels import NUM_CLASSES
from lora import apply_loras, lora_params_for_block

# One question per probe slot, low-level surface cues → high-level authorial semantics.
_PROBE_STRUCTURE_QUESTIONS: tuple[str, ...] = (
    "What punctuation marks and capitalization habits appear in this text?",
    "What suffixes and word endings recur in this passage?",
    "Which function words and fillers does the writer favor?",
    "How are subjects and verbs arranged in the sentences?",
    "How long and complex are the sentences on average?",
    "How do sentences link to one another with connectors?",
    "How are paragraphs and sections organized in this passage?",
    "What rhetorical patterns or figurative devices are used?",
    "What topics and semantic fields dominate this text?",
    "What overall authorial voice and style characterize this passage?",
)


def _unwrap_backbone(model: PreTrainedModel) -> PreTrainedModel:
    if hasattr(model, "get_base_model"):
        return model.get_base_model()
    return model


def _text_stack(model: PreTrainedModel) -> PreTrainedModel:
    model = _unwrap_backbone(model)
    if hasattr(model, "layers") and hasattr(model, "embed_tokens"):
        return model
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "layers") and hasattr(inner, "embed_tokens"):
        return inner
    language = getattr(model, "language_model", None)
    if language is not None:
        return _text_stack(language)
    raise TypeError(f"unsupported backbone module {type(model).__name__}")


def _backbone_layers(backbone: PreTrainedModel) -> nn.ModuleList:
    return _text_stack(backbone).layers


def _mean_question_embedding(
    question: str,
    tokenizer: PreTrainedTokenizerBase,
    embed_tokens: nn.Embedding,
) -> torch.Tensor:
    ids = tokenizer.encode(question, add_special_tokens=False)
    if not ids:
        raise ValueError(f"probe init question tokenizes to zero tokens: {question!r}")
    weight = embed_tokens.weight.data
    vecs = [weight[int(i)].clone() for i in ids]
    return torch.stack(vecs, dim=0).mean(dim=0)


def _probe_inits_from_questions(
    tokenizer: PreTrainedTokenizerBase,
    embed_tokens: nn.Embedding,
    *,
    num_probes: int,
) -> torch.Tensor:
    questions = _PROBE_STRUCTURE_QUESTIONS
    if len(questions) != num_probes:
        raise ValueError(
            f"num_probes={num_probes} but {len(questions)} probe init questions are defined"
        )
    rows = [_mean_question_embedding(q, tokenizer, embed_tokens) for q in questions]
    return torch.stack(rows, dim=0)


def _probe_token_hiddens(
    hidden: torch.Tensor,
    probe_mask: torch.Tensor,
    *,
    num_probes: int,
) -> torch.Tensor:
    """Last num_probes attended positions, fixed order (probe 0 … probe num_probes-1)."""
    lengths = probe_mask.sum(dim=1).long()
    batch = torch.arange(hidden.size(0), device=hidden.device)
    start = lengths - num_probes
    offsets = torch.arange(num_probes, device=hidden.device)
    idx = start.unsqueeze(1) + offsets.unsqueeze(0)
    picked = hidden[batch.unsqueeze(1), idx]
    return picked.reshape(hidden.size(0), num_probes * hidden.size(-1))


class ProbeClassifier(nn.Module):
    def __init__(
        self,
        backbone: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        *,
        tap_layers: tuple[int, ...],
        max_len: int,
        dropout: float,
        num_probes: int,
        contrast_hidden: int,
        contrast_out: int,
    ) -> None:
        super().__init__()
        stack = _text_stack(backbone)
        n_layers = int(stack.config.num_hidden_layers)
        hidden = int(stack.config.hidden_size)
        if num_probes < 1:
            raise ValueError("num_probes must be >= 1")
        for layer in tap_layers:
            if layer > n_layers:
                raise ValueError(f"tap layer {layer} is past the {n_layers} backbone blocks")
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.tap_layers = tap_layers
        self.max_len = max_len
        self.hidden_size = hidden
        self.num_probes = num_probes
        self.probes = nn.Parameter(
            _probe_inits_from_questions(
                tokenizer, stack.embed_tokens, num_probes=num_probes
            )
        )
        tap_width = num_probes * hidden
        self.pool_norms = nn.ModuleList(nn.LayerNorm(tap_width) for _ in tap_layers)
        self.dropout = nn.Dropout(dropout)
        pooled_dim = tap_width * len(tap_layers)
        self.classifier = nn.Linear(pooled_dim, NUM_CLASSES)
        self.contrast_head = nn.Sequential(
            nn.Linear(pooled_dim, contrast_hidden),
            nn.ReLU(),
            nn.Linear(contrast_hidden, contrast_out),
        )
        self._captured: dict[int, torch.Tensor] = {}
        self._hook_handles = []
        layers = _backbone_layers(backbone)
        for layer in tap_layers:
            handle = layers[layer - 1].register_forward_hook(self._make_hook(layer))
            self._hook_handles.append(handle)
        backbone.config.use_cache = False

    def _make_hook(self, layer: int):
        def _hook(_module, _inputs, output) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            self._captured[layer] = hidden

        return _hook

    def _embed_with_probes(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tok = _text_stack(self.backbone).embed_tokens(input_ids)
        batch, seq_len, hidden = tok.shape
        num = self.num_probes
        content_len = attention_mask.sum(dim=1).long()
        out_len = seq_len + num
        if out_len > self.max_len:
            raise ValueError(
                f"sequence length {out_len} exceeds max_len {self.max_len} after probes"
            )
        inputs = tok.new_zeros(batch, out_len, hidden)
        mask = attention_mask.new_zeros(batch, out_len)
        for row in range(batch):
            n = int(content_len[row])
            inputs[row, :n] = tok[row, :n]
            inputs[row, n : n + num] = self.probes
            tail = seq_len - n
            if tail:
                inputs[row, n + num : n + num + tail] = tok[row, n:seq_len]
            mask[row, : n + num] = 1
        return inputs, mask

    def _pooled_features(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        inputs_embeds, probe_mask = self._embed_with_probes(input_ids, attention_mask)
        self._captured.clear()
        self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=probe_mask,
            use_cache=False,
        )
        try:
            pooled = []
            for layer, norm in zip(self.tap_layers, self.pool_norms, strict=True):
                hidden = self._captured.get(layer)
                if hidden is None:
                    raise RuntimeError(f"block {layer} did not run")
                probes_flat = _probe_token_hiddens(
                    hidden, probe_mask, num_probes=self.num_probes
                )
                pooled.append(norm(probes_flat))
            return torch.cat(pooled, dim=-1)
        finally:
            self._captured.clear()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        pooled = self._pooled_features(input_ids, attention_mask)
        return self.classifier(self.dropout(pooled))

    def forward_logits_and_contrastive(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = self._pooled_features(input_ids, attention_mask)
        logits = self.classifier(self.dropout(pooled))
        contrast = F.normalize(self.contrast_head(pooled), dim=-1)
        return logits, contrast


def init_contrast_head(model: ProbeClassifier) -> None:
    for module in model.contrast_head.modules():
        reset = getattr(module, "reset_parameters", None)
        if callable(reset):
            reset()


def build_model(cfg: Config) -> ProbeClassifier:
    kw = hub_from_pretrained_kwargs(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, **kw)
    prepare_tokenizer(tokenizer, max_len=cfg.max_len)
    inner = _text_stack(
        AutoModel.from_pretrained(
            cfg.model_id,
            torch_dtype=torch.float32,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
            **kw,
        )
    )
    n_layers = int(inner.config.num_hidden_layers)
    backbone = apply_loras(inner, cfg.lora, num_layers=n_layers)
    model = ProbeClassifier(
        backbone,
        tokenizer,
        tap_layers=cfg.tap_layers,
        max_len=cfg.max_len,
        num_probes=cfg.probe.probe_count,
        dropout=cfg.head.dropout,
        contrast_hidden=cfg.contrast.hidden_dim,
        contrast_out=cfg.contrast.out_dim,
    )
    configure_trainability(model, cfg)
    init_contrast_head(model)
    return model


def configure_trainability(model: ProbeClassifier, cfg: Config) -> None:
    for param in model.backbone.parameters():
        param.requires_grad = False
    for block in cfg.lora.blocks:
        if not block.enabled:
            continue
        for param in lora_params_for_block(model.backbone, block):
            param.requires_grad = True
    model.probes.requires_grad = not cfg.probe.freeze
    for param in model.pool_norms.parameters():
        param.requires_grad = not cfg.head.freeze
    for param in model.classifier.parameters():
        param.requires_grad = not cfg.head.freeze
    for param in model.contrast_head.parameters():
        param.requires_grad = not cfg.contrast.freeze
