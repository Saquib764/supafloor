"""Save and load decision-model checkpoints (encoder, tokenizer, score head)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch

from config import (
    ARCHITECTURE_MASK_DECISION,
    ARCHITECTURE_MASK_TOKEN,
    ARCHITECTURE_MEAN_POOL,
    CHECKPOINT_ARCHITECTURES,
    DecisionConfig,
    hub_from_pretrained_kwargs,
    load_config,
)
from formatting import resolve_sequence_length
from model.decision_model import ModernBERTDecisionModel, configure_for_training

CHECKPOINT_ENCODER_DIR = "encoder"
CHECKPOINT_TOKENIZER_DIR = "tokenizer"
CHECKPOINT_SCORER = "scoring_head.pt"
CHECKPOINT_DECISION_STACK = "decision_stack.pt"
CHECKPOINT_META = "decision_meta.json"
CHECKPOINT_KIND_FULL = "full"
LEGACY_LORA_DIR = "lora_adapter"


def encoder_checkpoint_present(root: Path) -> bool:
    encoder_dir = root / CHECKPOINT_ENCODER_DIR
    return encoder_dir.is_dir() and (encoder_dir / "config.json").is_file()


def _legacy_lora_checkpoint(root: Path, meta: dict[str, object]) -> bool:
    if bool(meta.get("lora_enabled")):
        return True
    lora_dir = root / LEGACY_LORA_DIR
    return lora_dir.is_dir() and (lora_dir / "adapter_config.json").is_file()


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

    model.tokenizer.save_pretrained(tokenizer_dir)
    torch.save(model.score_head.state_dict(), root / CHECKPOINT_SCORER)
    torch.save(model.decision_stack.state_dict(), root / CHECKPOINT_DECISION_STACK)

    if write_encoder:
        _atomic_save_pretrained(model.encoder, encoder_dir)
        print(f"checkpoint: saved encoder → {encoder_dir}", flush=True)

    dl = model.cfg.model.decision_layers
    meta = {
        "architecture": ARCHITECTURE_MASK_DECISION,
        "score_token": model.cfg.model.score_head.token,
        "score_head_hidden_size": model.cfg.model.score_head.hidden_size,
        "checkpoint_kind": CHECKPOINT_KIND_FULL,
        "pretrained_name": model.cfg.pretrained_name,
        "max_len": model.max_len,
        "hidden_size": model.hidden_size,
        "base_vocab_size": model.base_vocab_size,
        "decision_layers": {
            "count": dl.count,
            "hidden_size": dl.hidden_size,
            "num_attention_heads": dl.num_attention_heads,
            "intermediate_size": dl.intermediate_size,
            "pre_norm": dl.pre_norm,
            "dropout": dl.dropout,
        },
        "finetune_last_n_layers": model.finetune_last_n_layers,
        "encoder_layer_train": list(model.encoder_layer_train),
        "encoder_layer_learning_rates": list(model.encoder_finetune_lrs),
    }
    meta_tmp = _meta_path(root).with_suffix(".json.tmp")
    meta_tmp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    meta_tmp.replace(_meta_path(root))


def _load_encoder_from_hub(cfg: DecisionConfig):
    from transformers import AutoModel

    return AutoModel.from_pretrained(cfg.pretrained_name, **hub_from_pretrained_kwargs(cfg))


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
    n = int(meta["finetune_last_n_layers"])
    train = tuple(bool(x) for x in meta["encoder_layer_train"])  # type: ignore[union-attr]
    lrs = tuple(float(x) for x in meta["encoder_layer_learning_rates"])  # type: ignore[union-attr]
    if len(train) != n or len(lrs) != n:
        raise ValueError("checkpoint finetune meta lists length mismatch")
    return n, train, lrs


def _assert_decision_layers_match(meta_layers: object, cfg: DecisionConfig) -> None:
    if meta_layers is None:
        return
    dl = cfg.model.decision_layers
    if isinstance(meta_layers, int):
        if meta_layers != dl.count:
            raise ValueError(
                f"checkpoint decision_layers count {meta_layers} != config {dl.count}"
            )
        return
    if not isinstance(meta_layers, dict):
        raise ValueError("checkpoint decision_layers meta must be a mapping or int")
    checks: tuple[tuple[str, object], ...] = (
        ("count", dl.count),
        ("hidden_size", dl.hidden_size),
        ("num_attention_heads", dl.num_attention_heads),
        ("intermediate_size", dl.intermediate_size),
        ("pre_norm", dl.pre_norm),
        ("dropout", dl.dropout),
    )
    for key, expected in checks:
        if key not in meta_layers:
            continue
        got = meta_layers[key]
        if key == "dropout":
            if abs(float(got) - float(expected)) > 1e-9:
                raise ValueError(f"checkpoint decision_layers.{key} {got} != config {expected}")
        elif key == "pre_norm":
            if bool(got) != bool(expected):
                raise ValueError(f"checkpoint decision_layers.{key} {got!r} != config {expected!r}")
        elif int(got) != int(expected):
            raise ValueError(f"checkpoint decision_layers.{key} {got!r} != config {expected!r}")


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
    if _legacy_lora_checkpoint(root, meta):
        raise ValueError(
            "checkpoint uses LoRA adapters, which are no longer supported; "
            "retrain with full encoder fine-tuning or export merged weights"
        )
    arch = str(meta.get("architecture") or ARCHITECTURE_MEAN_POOL)
    if arch == ARCHITECTURE_MEAN_POOL:
        raise ValueError(
            f"checkpoint uses legacy architecture {arch!r}; retrain with "
            f"{ARCHITECTURE_MASK_TOKEN!r} (mask-token score head)"
        )
    if arch not in CHECKPOINT_ARCHITECTURES:
        raise ValueError(f"unsupported checkpoint architecture {arch!r}")
    if arch == ARCHITECTURE_MASK_TOKEN:
        raise ValueError(
            f"checkpoint uses legacy architecture {arch!r}; retrain with "
            f"{ARCHITECTURE_MASK_DECISION!r} (decision stack over option hiddens)"
        )
    cfg = cfg or load_config()
    _assert_decision_layers_match(meta.get("decision_layers"), cfg)
    meta_score = meta.get("score_token")
    if meta_score is not None and str(meta_score) != cfg.model.score_head.token:
        raise ValueError(
            f"checkpoint score_token {meta_score!r} != config model.score_head.token "
            f"{cfg.model.score_head.token!r}"
        )
    meta_sh = meta.get("score_head_hidden_size")
    if meta_sh is not None and int(meta_sh) != cfg.model.score_head.hidden_size:
        raise ValueError(
            f"checkpoint score_head_hidden_size {meta_sh} != config "
            f"{cfg.model.score_head.hidden_size}"
        )
    if meta.get("pretrained_name") and meta["pretrained_name"] != cfg.pretrained_name:
        raise ValueError(
            f"checkpoint pretrained_name {meta['pretrained_name']!r} "
            f"does not match config {cfg.pretrained_name!r}"
        )
    from transformers import AutoTokenizer

    encoder_dir = root / CHECKPOINT_ENCODER_DIR
    tokenizer_dir = root / CHECKPOINT_TOKENIZER_DIR
    scorer_path = root / CHECKPOINT_SCORER
    decision_path = root / CHECKPOINT_DECISION_STACK
    if not tokenizer_dir.is_dir():
        raise FileNotFoundError(f"missing tokenizer: {tokenizer_dir}")
    if not scorer_path.is_file():
        raise FileNotFoundError(f"missing scoring head: {scorer_path}")
    if not decision_path.is_file():
        raise FileNotFoundError(f"missing decision stack: {decision_path}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.sep_token is not None:
            tokenizer.pad_token = tokenizer.sep_token

    saved_encoder = encoder_checkpoint_present(root)
    if saved_encoder:
        from transformers import AutoModel

        encoder = AutoModel.from_pretrained(encoder_dir)
        print(f"checkpoint: loaded encoder from {encoder_dir}", flush=True)
    else:
        encoder = _load_encoder_from_hub(cfg)
        print(
            f"checkpoint: loaded base encoder from hub ({cfg.pretrained_name})",
            flush=True,
        )

    max_len = resolve_sequence_length(tokenizer, encoder.config, cap=cfg.model.max_len)
    meta_len = meta.get("max_len")
    if meta_len is not None and int(meta_len) != max_len:
        raise ValueError(
            f"checkpoint max_len {meta_len} != config model.max_len {max_len}; refusing to load"
        )
    model = ModernBERTDecisionModel(
        encoder,
        tokenizer,
        cfg=cfg,
        max_len=max_len,
    )
    state = torch.load(scorer_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and state.keys() == {"weight"}:
        raise ValueError(
            "checkpoint scoring_head.pt uses legacy single Linear weights; retrain with MLP score head"
        )
    model.score_head.load_state_dict(state, strict=True)
    stack_state = torch.load(decision_path, map_location="cpu", weights_only=False)
    model.decision_stack.load_state_dict(stack_state, strict=True)
    _finalize_trainability(
        model,
        cfg,
        meta,
        trainability_from_config=trainability_from_config,
    )
    return model
