"""Save and load decision-model checkpoints (encoder, tokenizer, score head)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch

from config import (
    ARCHITECTURE_CLS_PER_CANDIDATE,
    ARCHITECTURE_MEAN_POOL,
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
CHECKPOINT_META = "decision_meta.json"
CHECKPOINT_KIND_FULL = "full"
CHECKPOINT_KIND_LORA = "lora"


def _log_lora(message: str) -> None:
    print(f"LoRA: {message}", flush=True)


def lora_adapter_present(root: Path) -> bool:
    lora_dir = root / CHECKPOINT_LORA_DIR
    return lora_dir.is_dir() and (lora_dir / "adapter_config.json").is_file()


def encoder_checkpoint_present(root: Path) -> bool:
    encoder_dir = root / CHECKPOINT_ENCODER_DIR
    return encoder_dir.is_dir() and (encoder_dir / "config.json").is_file()


def _meta_path(root: Path) -> Path:
    return root / CHECKPOINT_META


def _atomic_save_pretrained(module, dest_dir: Path) -> None:
    dest_dir = dest_dir.resolve()
    tmp_dir = dest_dir.with_name(f"{dest_dir.name}.tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    try:
        module.save_pretrained(tmp_dir)
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        tmp_dir.rename(dest_dir)
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def _encoder_for_full_save(model: ModernBERTDecisionModel):
    enc = model.encoder
    get_base = getattr(enc, "get_base_model", None)
    if get_base is not None:
        return get_base()
    return enc


def _atomic_save_encoder(model: ModernBERTDecisionModel, encoder_dir: Path) -> None:
    _atomic_save_pretrained(_encoder_for_full_save(model), encoder_dir)


def save_checkpoint(
    model: ModernBERTDecisionModel,
    root: Path,
    *,
    write_encoder: bool = True,
    at_end: bool = False,
) -> None:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = root / CHECKPOINT_TOKENIZER_DIR
    encoder_dir = root / CHECKPOINT_ENCODER_DIR
    lora = model.cfg.train.lora
    use_lora = model.lora_enabled and lora.enabled

    model.tokenizer.save_pretrained(tokenizer_dir)
    torch.save({"weight": model.score_head.weight.detach().cpu()}, root / CHECKPOINT_SCORER)

    if write_encoder:
        _atomic_save_encoder(model, encoder_dir)
        print(f"checkpoint: saved full encoder → {encoder_dir}", flush=True)

    if use_lora:
        from model.lora import save_encoder_lora_adapter

        lora_dir = root / CHECKPOINT_LORA_DIR
        save_encoder_lora_adapter(model.encoder, lora_dir)
        print(f"checkpoint: saved LoRA adapter → {lora_dir}", flush=True)

    kind = CHECKPOINT_KIND_LORA if use_lora else CHECKPOINT_KIND_FULL

    meta = {
        "architecture": ARCHITECTURE_MEAN_POOL,
        "checkpoint_kind": kind,
        "pretrained_name": model.cfg.pretrained_name,
        "max_len": model.max_len,
        "hidden_size": model.hidden_size,
        "base_vocab_size": model.base_vocab_size,
        "finetune_last_n_layers": model.finetune_last_n_layers,
        "encoder_layer_train": list(model.encoder_layer_train),
        "encoder_layer_learning_rates": list(model.encoder_finetune_lrs),
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
    if arch == ARCHITECTURE_MEAN_POOL:
        return
    if arch == ARCHITECTURE_CLS_PER_CANDIDATE:
        raise ValueError(
            "checkpoint uses CLS scoring (cls_per_candidate_v1); retrain with masked mean pooling"
        )
    if arch == ARCHITECTURE_SEGMENT_MARKERS:
        raise ValueError(
            "checkpoint uses learnable segment marker tokens; retrain from Hub base"
        )
    if meta.get("marker_tokens") or meta.get("marker_ids") or meta.get("marker_embedding_train"):
        raise ValueError(
            "checkpoint uses an obsolete marker layout; retrain from Hub base "
            f"(expected architecture {ARCHITECTURE_MEAN_POOL!r})"
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
) -> tuple[int, tuple[bool, ...], tuple[float, ...]]:
    if trainability_from_config or "finetune_last_n_layers" not in meta:
        return (
            cfg.model.finetune_last_n_layers,
            cfg.train.encoder_layer_train,
            cfg.train.encoder_layer_learning_rates,
        )
    _reject_legacy_full_finetune(meta)
    n = int(meta["finetune_last_n_layers"])
    train = tuple(bool(x) for x in meta["encoder_layer_train"])  # type: ignore[union-attr]
    lrs = tuple(float(x) for x in meta["encoder_layer_learning_rates"])  # type: ignore[union-attr]
    if len(train) != n or len(lrs) != n:
        raise ValueError("checkpoint finetune meta lists length mismatch")
    return n, train, lrs


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
        )
        return
    finetune_n, layer_train, layer_lrs = _resolve_encoder_trainability(
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

    lora_in_meta = bool(meta.get("lora_enabled"))
    adapter_present = lora_adapter_present(root)
    saved_encoder = encoder_checkpoint_present(root)
    if saved_encoder:
        from transformers import AutoModel

        encoder = AutoModel.from_pretrained(encoder_dir)
        print(f"checkpoint: loaded full encoder from {encoder_dir}", flush=True)
    else:
        encoder = _load_encoder_from_hub(cfg)
        print(
            f"checkpoint: loaded base encoder from hub ({cfg.pretrained_name})",
            flush=True,
        )

    lora_loaded = False
    if adapter_present:
        from peft import PeftModel

        from model.lora import load_encoder_lora_adapter

        if isinstance(encoder, PeftModel):
            raise ValueError("encoder is already a PeftModel before loading adapter")
        encoder = load_encoder_lora_adapter(encoder, lora_dir)
        lora_loaded = True
        _log_lora(f"loaded adapter from {lora_dir}")
        if not lora_in_meta:
            _log_lora("adapter on disk but decision_meta lora_enabled=false (loaded anyway)")
    elif lora_in_meta:
        raise FileNotFoundError(f"missing LoRA adapter: {lora_dir}")
    else:
        _log_lora("no adapter in checkpoint directory")

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
    model.lora_enabled = lora_loaded
    state = torch.load(scorer_path, map_location="cpu", weights_only=False)
    model.score_head.load_state_dict(state, strict=True)
    if lora_loaded:
        from model.lora import lora_trainable_parameter_count

        _log_lora(f"active ({lora_trainable_parameter_count(model.encoder):,} adapter params)")
    _finalize_trainability(
        model,
        cfg,
        meta,
        trainability_from_config=trainability_from_config,
    )
    return model
