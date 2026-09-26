"""Save and load decision-model checkpoints (encoder, tokenizer, scorer, markers)."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path

import torch

from config import DecisionConfig, MarkerTokenSettings, hub_from_pretrained_kwargs, load_config
from formatting import MarkerIds, marker_ids_from_tokenizer, max_sequence_length
from model.decision_model import ModernBERTDecisionModel

CHECKPOINT_ENCODER_DIR = "encoder"
CHECKPOINT_TOKENIZER_DIR = "tokenizer"
CHECKPOINT_SCORER = "scoring_head.pt"
CHECKPOINT_MARKER_EMBEDDINGS = "marker_embeddings.pt"
CHECKPOINT_META = "decision_meta.json"
CHECKPOINT_KIND_LIGHT = "light"
CHECKPOINT_KIND_FULL = "full"


def _meta_path(root: Path) -> Path:
    return root / CHECKPOINT_META


def _marker_embedding_state(model: ModernBERTDecisionModel) -> dict[str, object]:
    embed = model.encoder.get_input_embeddings()
    if embed is None:
        raise RuntimeError("encoder has no input embeddings")
    indices = [
        int(model.markers.state),
        int(model.markers.question),
        int(model.markers.option),
    ]
    rows = embed.weight.detach().cpu()[indices]
    return {"indices": indices, "weight": rows}


def _apply_marker_embedding_state(model: ModernBERTDecisionModel, state: dict[str, object]) -> None:
    embed = model.encoder.get_input_embeddings()
    if embed is None:
        raise RuntimeError("encoder has no input embeddings")
    indices = [int(i) for i in state["indices"]]  # type: ignore[index]
    rows = state["weight"]  # type: ignore[assignment]
    if not isinstance(rows, torch.Tensor):
        raise ValueError("marker_embeddings weight must be a tensor")
    if len(indices) != rows.shape[0]:
        raise ValueError("marker_embeddings indices/weight shape mismatch")
    device = embed.weight.device
    dtype = embed.weight.dtype
    for row_index, token_id in enumerate(indices):
        embed.weight.data[token_id] = rows[row_index].to(device=device, dtype=dtype)


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


def should_write_encoder(cfg: DecisionConfig, *, at_end: bool) -> bool:
    if cfg.model.train_pretrained_encoder:
        return at_end or cfg.train.save_full_encoder
    return bool(cfg.train.save_full_encoder and at_end)


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
    if write_encoder is None:
        write_encoder = should_write_encoder(model.cfg, at_end=at_end)

    model.tokenizer.save_pretrained(tokenizer_dir)
    torch.save({"weight": model.score_head.weight.detach().cpu()}, root / CHECKPOINT_SCORER)
    torch.save(_marker_embedding_state(model), root / CHECKPOINT_MARKER_EMBEDDINGS)

    kind = CHECKPOINT_KIND_FULL if write_encoder else CHECKPOINT_KIND_LIGHT
    if write_encoder:
        _atomic_save_encoder(model, encoder_dir)
    elif encoder_dir.is_dir() and kind == CHECKPOINT_KIND_LIGHT:
        pass

    meta = {
        "checkpoint_kind": kind,
        "pretrained_name": model.cfg.pretrained_name,
        "max_len": model.max_len,
        "hidden_size": model.hidden_size,
        "base_vocab_size": model.base_vocab_size,
        "train_pretrained_encoder": model.train_pretrained_encoder,
        "marker_tokens": asdict(model.cfg.marker_tokens),
        "marker_ids": asdict(model.markers),
    }
    meta_tmp = _meta_path(root).with_suffix(".json.tmp")
    meta_tmp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    meta_tmp.replace(_meta_path(root))


def _load_encoder_from_hub(cfg: DecisionConfig):
    from transformers import AutoModel

    return AutoModel.from_pretrained(cfg.pretrained_name, **hub_from_pretrained_kwargs(cfg))


def load_checkpoint(root: Path, cfg: DecisionConfig | None = None) -> ModernBERTDecisionModel:
    root = root.resolve()
    meta_file = _meta_path(root)
    if not meta_file.is_file():
        raise FileNotFoundError(f"missing checkpoint metadata: {meta_file}")
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    cfg = cfg or load_config()
    if meta.get("pretrained_name") and meta["pretrained_name"] != cfg.pretrained_name:
        raise ValueError(
            f"checkpoint pretrained_name {meta['pretrained_name']!r} "
            f"does not match config {cfg.pretrained_name!r}"
        )
    marker_tokens = MarkerTokenSettings(**meta["marker_tokens"])
    cfg_markers = cfg.marker_tokens
    if (
        marker_tokens.state != cfg_markers.state
        or marker_tokens.question != cfg_markers.question
        or marker_tokens.option != cfg_markers.option
    ):
        raise ValueError("checkpoint marker_tokens do not match config marker_tokens")

    from transformers import AutoModel, AutoTokenizer

    encoder_dir = root / CHECKPOINT_ENCODER_DIR
    tokenizer_dir = root / CHECKPOINT_TOKENIZER_DIR
    scorer_path = root / CHECKPOINT_SCORER
    marker_path = root / CHECKPOINT_MARKER_EMBEDDINGS
    if not tokenizer_dir.is_dir():
        raise FileNotFoundError(f"missing tokenizer: {tokenizer_dir}")
    if not scorer_path.is_file():
        raise FileNotFoundError(f"missing scoring head: {scorer_path}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)
    kind = str(meta.get("checkpoint_kind") or CHECKPOINT_KIND_FULL)
    use_local_encoder = kind == CHECKPOINT_KIND_FULL and encoder_dir.is_dir()
    if use_local_encoder:
        encoder = AutoModel.from_pretrained(encoder_dir)
    else:
        encoder = _load_encoder_from_hub(cfg)
        encoder.resize_token_embeddings(len(tokenizer))

    markers = marker_ids_from_tokenizer(tokenizer, cfg)
    stored_ids = meta.get("marker_ids") or {}
    for key in ("state", "question", "option"):
        if int(stored_ids.get(key, -1)) != getattr(markers, key):
            raise ValueError(f"marker id mismatch for {key!r} after reload")

    max_len = int(meta.get("max_len") or max_sequence_length(tokenizer, encoder.config))
    model = ModernBERTDecisionModel(
        encoder,
        tokenizer,
        cfg=cfg,
        markers=markers,
        max_len=max_len,
    )
    state = torch.load(scorer_path, map_location="cpu", weights_only=False)
    model.score_head.load_state_dict(state, strict=True)
    if marker_path.is_file():
        marker_state = torch.load(marker_path, map_location="cpu", weights_only=False)
        _apply_marker_embedding_state(model, marker_state)
    base_vocab = int(meta.get("base_vocab_size") or 0)
    if base_vocab <= 0:
        raise ValueError("checkpoint missing valid base_vocab_size")
    if "train_pretrained_encoder" in meta:
        train_encoder = bool(meta["train_pretrained_encoder"])
    elif meta.get("freeze_encoder") is not None:
        train_encoder = not bool(meta["freeze_encoder"])
    else:
        train_encoder = cfg.model.train_pretrained_encoder
    from model.freeze import apply_encoder_trainability

    apply_encoder_trainability(
        model,
        base_vocab_size=base_vocab,
        train_pretrained_encoder=train_encoder,
    )
    return model
