"""Save and load decision-model checkpoints (encoder, tokenizer, CLS scorer)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch

from config import (
    ARCHITECTURE_CLS_PER_CANDIDATE,
    ARCHITECTURE_SEGMENT_MARKERS,
    DecisionConfig,
    hub_from_pretrained_kwargs,
    load_config,
)
from formatting import max_sequence_length
from model.decision_model import ModernBERTDecisionModel, configure_for_training

CHECKPOINT_ENCODER_DIR = "encoder"
CHECKPOINT_TOKENIZER_DIR = "tokenizer"
CHECKPOINT_LORA_DIR = "lora_adapter"
CHECKPOINT_SCORER = "scoring_head.pt"
CHECKPOINT_CLS_EMBEDDING = "cls_embedding.pt"
CHECKPOINT_META = "decision_meta.json"
CHECKPOINT_KIND_FULL = "full"
CHECKPOINT_KIND_LORA = "lora"


def _meta_path(root: Path) -> Path:
    return root / CHECKPOINT_META


def _cls_embedding_state(model: ModernBERTDecisionModel) -> dict[str, object] | None:
    if model.cls_token_id is None:
        return None
    embed = model.encoder.get_input_embeddings()
    if embed is None:
        return None
    token_id = int(model.cls_token_id)
    return {"index": token_id, "weight": embed.weight.detach().cpu()[token_id]}


def _apply_cls_embedding_state(model: ModernBERTDecisionModel, state: dict[str, object]) -> None:
    embed = model.encoder.get_input_embeddings()
    if embed is None:
        raise RuntimeError("encoder has no input embeddings")
    token_id = int(state["index"])  # type: ignore[arg-type]
    row = state["weight"]  # type: ignore[assignment]
    if not isinstance(row, torch.Tensor):
        raise ValueError("cls_embedding weight must be a tensor")
    device = embed.weight.device
    dtype = embed.weight.dtype
    embed.weight.data[token_id] = row.to(device=device, dtype=dtype)


def _atomic_save_encoder(model: ModernBERTDecisionModel, encoder_dir: Path) -> None:
    encoder_dir = encoder_dir.resolve()
    tmp_dir = encoder_dir.with_name(f"{encoder_dir.name}.tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    try:
        model.encoder.save_pretrained(tmp_dir)
        if encoder_dir.exists():
            shutil.rmtree(encoder_dir)
        tmp_dir.rename(encoder_dir)
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def save_checkpoint(
    model: ModernBERTDecisionModel,
    root: Path,
    *,
    write_encoder: bool | None = None,
    at_end: bool = False,
) -> None:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = root / CHECKPOINT_TOKENIZER_DIR
    encoder_dir = root / CHECKPOINT_ENCODER_DIR
    lora = model.cfg.train.lora
    use_lora = model.lora_enabled and lora.enabled
    if write_encoder is None:
        write_encoder = not use_lora

    model.tokenizer.save_pretrained(tokenizer_dir)
    torch.save({"weight": model.score_head.weight.detach().cpu()}, root / CHECKPOINT_SCORER)
    cls_state = _cls_embedding_state(model)
    if cls_state is not None:
        torch.save(cls_state, root / CHECKPOINT_CLS_EMBEDDING)

    kind = CHECKPOINT_KIND_LORA if use_lora else CHECKPOINT_KIND_FULL
    if use_lora:
        from model.lora import save_encoder_lora_adapter

        save_encoder_lora_adapter(model.encoder, root / CHECKPOINT_LORA_DIR)
    elif write_encoder:
        _atomic_save_encoder(model, encoder_dir)

    meta = {
        "architecture": ARCHITECTURE_CLS_PER_CANDIDATE,
        "checkpoint_kind": kind,
        "pretrained_name": model.cfg.pretrained_name,
        "max_len": model.max_len,
        "hidden_size": model.hidden_size,
        "base_vocab_size": model.base_vocab_size,
        "finetune_last_n_layers": model.finetune_last_n_layers,
        "encoder_layer_train": list(model.encoder_layer_train),
        "encoder_layer_learning_rates": list(model.encoder_finetune_lrs),
        "train_cls_embedding": model.train_cls_embedding,
        "lora_enabled": model.lora_enabled,
        "lora_rank": lora.rank if use_lora else None,
        "lora_alpha": lora.alpha if use_lora else None,
        "lora_dropout": lora.dropout if use_lora else None,
    }
    meta_tmp = _meta_path(root).with_suffix(".json.tmp")
    meta_tmp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    meta_tmp.replace(_meta_path(root))


def _load_encoder_from_hub(cfg: DecisionConfig):
    from transformers import AutoModel

    return AutoModel.from_pretrained(cfg.pretrained_name, **hub_from_pretrained_kwargs(cfg))


def _reject_legacy_checkpoint(meta: dict[str, object]) -> None:
    arch = meta.get("architecture")
    if arch == ARCHITECTURE_CLS_PER_CANDIDATE:
        return
    if arch == ARCHITECTURE_SEGMENT_MARKERS:
        raise ValueError(
            "checkpoint uses learnable segment marker tokens; retrain from Hub base with CLS+[SEP] layout"
        )
    if meta.get("marker_tokens") or meta.get("marker_ids") or meta.get("marker_embedding_train"):
        raise ValueError(
            "checkpoint uses an obsolete marker layout; retrain from Hub base "
            f"(expected architecture {ARCHITECTURE_CLS_PER_CANDIDATE!r})"
        )


def _reject_legacy_full_finetune(meta: dict[str, object]) -> None:
    if bool(meta.get("train_pretrained_encoder", False)):
        raise ValueError(
            "checkpoint used removed full-encoder fine-tune (train_pretrained_encoder); retrain"
        )
    if meta.get("freeze_encoder") is False:
        raise ValueError("checkpoint used removed full-encoder fine-tune (freeze_encoder: false)")


def _resolve_encoder_trainability(
    meta: dict[str, object],
    cfg: DecisionConfig,
    *,
    trainability_from_config: bool,
) -> tuple[int, tuple[bool, ...], tuple[float, ...], bool]:
    if trainability_from_config or "finetune_last_n_layers" not in meta:
        return (
            cfg.model.finetune_last_n_layers,
            cfg.train.encoder_layer_train,
            cfg.train.encoder_layer_learning_rates,
            cfg.train.train_cls_embedding,
        )
    _reject_legacy_full_finetune(meta)
    n = int(meta["finetune_last_n_layers"])
    train = tuple(bool(x) for x in meta["encoder_layer_train"])  # type: ignore[union-attr]
    lrs = tuple(float(x) for x in meta["encoder_layer_learning_rates"])  # type: ignore[union-attr]
    if len(train) != n or len(lrs) != n:
        raise ValueError("checkpoint finetune meta lists length mismatch")
    train_cls = (
        bool(meta["train_cls_embedding"])
        if "train_cls_embedding" in meta
        else cfg.train.train_cls_embedding
    )
    return n, train, lrs, train_cls


def _finalize_trainability(
    model: ModernBERTDecisionModel,
    cfg: DecisionConfig,
    meta: dict[str, object],
    *,
    trainability_from_config: bool,
) -> None:
    if trainability_from_config:
        configure_for_training(model, cfg)
        return
    if bool(meta.get("lora_enabled")):
        if not model.lora_enabled:
            raise ValueError("checkpoint metadata expects LoRA but adapter was not loaded")
        from model.freeze import apply_encoder_trainability

        apply_encoder_trainability(
            model,
            finetune_last_n_layers=0,
            encoder_layer_train=(),
            encoder_layer_learning_rates=(),
            train_cls_embedding=bool(meta.get("train_cls_embedding", cfg.train.train_cls_embedding)),
        )
        return
    finetune_n, layer_train, layer_lrs, train_cls = _resolve_encoder_trainability(
        meta,
        cfg,
        trainability_from_config=False,
    )
    from model.freeze import apply_encoder_trainability

    apply_encoder_trainability(
        model,
        finetune_last_n_layers=finetune_n,
        encoder_layer_train=layer_train,
        encoder_layer_learning_rates=layer_lrs,
        train_cls_embedding=train_cls,
    )


def load_checkpoint(
    root: Path,
    cfg: DecisionConfig | None = None,
    *,
    trainability_from_config: bool = False,
) -> ModernBERTDecisionModel:
    root = root.resolve()
    meta_file = _meta_path(root)
    if not meta_file.is_file():
        raise FileNotFoundError(f"missing checkpoint metadata: {meta_file}")
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    _reject_legacy_checkpoint(meta)
    cfg = cfg or load_config()
    if meta.get("pretrained_name") and meta["pretrained_name"] != cfg.pretrained_name:
        raise ValueError(
            f"checkpoint pretrained_name {meta['pretrained_name']!r} "
            f"does not match config {cfg.pretrained_name!r}"
        )
    from transformers import AutoTokenizer

    encoder_dir = root / CHECKPOINT_ENCODER_DIR
    tokenizer_dir = root / CHECKPOINT_TOKENIZER_DIR
    lora_dir = root / CHECKPOINT_LORA_DIR
    scorer_path = root / CHECKPOINT_SCORER
    if not tokenizer_dir.is_dir():
        raise FileNotFoundError(f"missing tokenizer: {tokenizer_dir}")
    if not scorer_path.is_file():
        raise FileNotFoundError(f"missing scoring head: {scorer_path}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.sep_token is not None:
            tokenizer.pad_token = tokenizer.sep_token

    lora_in_ckpt = bool(meta.get("lora_enabled"))
    kind = str(meta.get("checkpoint_kind") or CHECKPOINT_KIND_FULL)
    use_local_encoder = (
        not lora_in_ckpt
        and kind == CHECKPOINT_KIND_FULL
        and encoder_dir.is_dir()
    )
    if use_local_encoder:
        from transformers import AutoModel

        encoder = AutoModel.from_pretrained(encoder_dir)
    else:
        encoder = _load_encoder_from_hub(cfg)

    if lora_in_ckpt:
        if not lora_dir.is_dir():
            raise FileNotFoundError(f"missing LoRA adapter: {lora_dir}")
        from model.lora import load_encoder_lora_adapter

        encoder = load_encoder_lora_adapter(encoder, lora_dir)

    max_len = max_sequence_length(tokenizer, encoder.config)
    meta_len = meta.get("max_len")
    if meta_len is not None and int(meta_len) != max_len:
        raise ValueError(
            f"checkpoint max_len {meta_len} != encoder context {max_len}; refusing to load"
        )
    model = ModernBERTDecisionModel(
        encoder,
        tokenizer,
        cfg=cfg,
        max_len=max_len,
    )
    model.lora_enabled = lora_in_ckpt
    state = torch.load(scorer_path, map_location="cpu", weights_only=False)
    model.score_head.load_state_dict(state, strict=True)
    cls_path = root / CHECKPOINT_CLS_EMBEDDING
    if cls_path.is_file():
        cls_state = torch.load(cls_path, map_location="cpu", weights_only=False)
        _apply_cls_embedding_state(model, cls_state)
    _finalize_trainability(
        model,
        cfg,
        meta,
        trainability_from_config=trainability_from_config,
    )
    return model
