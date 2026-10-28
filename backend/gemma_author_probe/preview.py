"""Print model summary: frozen Qwen3, probe tokens, multi-layer head."""

from __future__ import annotations

import torch

from config import Config, load_config
from device import get_device
from labels import MODEL_NAMES, NUM_CLASSES
from lora import lora_params_for_block
from model import ProbeClassifier, build_model


def _param_counts(module: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def _describe_head(cfg: Config) -> list[str]:
    return [
        "type:           LayerNorm per block, concat, dropout, Linear",
        f"tap_layers:     {list(cfg.tap_layers)}",
        f"dropout:        {cfg.head.dropout:g}",
        f"num_classes:    {NUM_CLASSES}",
        f"probe tokens:   {cfg.probe.probe_count} (fixed order, concat hiddens per tap)",
        "forward:        all probe hiddens at each tap → LN → concat taps → dropout → Linear",
        "probe init:     each probe = mean token embed of one structure question (low→semantic)",
        "loss:           cross entropy (+ optional contrastive on contrast head)",
        f"contrast head:  Linear→ReLU→Linear→L2 "
        f"({cfg.contrast.hidden_dim}→{cfg.contrast.out_dim})",
    ]


def run(
    cfg: Config | None = None,
    *,
    device: torch.device | None = None,
    build_from_hub: bool = True,
) -> ProbeClassifier | None:
    cfg = cfg or load_config()
    device = device or get_device(announce=True)
    train = cfg.train
    n_taps = len(cfg.tap_layers)

    print("\n=== Name That LLM (Qwen3 blocks + probe + 18-class head) ===\n")
    print(f"model_id:       {cfg.model_id}")
    print(
        f"max_len:        {cfg.max_len}  "
        f"(tokenize to {cfg.max_len - cfg.probe.probe_count}, append {cfg.probe.probe_count} probes)"
    )
    print(f"probe.probe_count: {cfg.probe.probe_count}")
    print(f"prompt_count:      {train.prompt_count}")
    print(
        f"authors per batch: {train.authors_per_batch}  "
        f"(train.batch_size / train.prompt_count)"
    )
    print(f"tap_layers:     {list(cfg.tap_layers)}  (1-indexed blocks)")
    print(
        f"input:          passage + {cfg.probe.probe_count} probes; "
        f"concat probe hiddens at {n_taps} taps"
    )
    print(f"output:         logits [{NUM_CLASSES}]")

    print("\n=== Head ===\n")
    for line in _describe_head(cfg):
        print(line)

    print("\n=== Labels ===\n")
    for i, name in enumerate(MODEL_NAMES):
        print(f"  {i:2d}  {name}")

    print("\n=== Training (config) ===\n")
    print("embeddings:       frozen")
    for block in cfg.lora.blocks:
        if block.enabled:
            print(
                f"{block.name}:         layers={list(block.layers)}  "
                f"r={block.r} lr={block.lr:g}"
            )
        else:
            print(f"{block.name}:         off")
    print(f"probe freeze:     {cfg.probe.freeze}")
    if not cfg.probe.freeze:
        print(f"probe lr:         {cfg.probe.lr:g}")
    print(f"head freeze:      {cfg.head.freeze}")
    if not cfg.head.freeze:
        print(f"head lr:          {cfg.head.lr:g}  (pool norms + linear)")
    print(f"contrast freeze:  {cfg.contrast.freeze}")
    if not cfg.contrast.freeze:
        print(f"contrast lr:      {cfg.contrast.lr:g}")
    print(f"contrast weight:  {cfg.contrast.weight:g}")
    print(
        f"loss:             cross entropy "
        f"(batch={train.batch_size}: {train.prompt_count} prompts × "
        f"{train.authors_per_batch} authors)"
    )

    if not build_from_hub:
        print("\n(skipped hub load; use full preview for parameter counts)\n")
        return None

    print("\n=== Loading weights from hub (parameter counts) ===\n", flush=True)
    model = build_model(cfg)

    back_total, back_train = _param_counts(model.backbone)
    probe_total = int(model.probes.numel())
    norm_total, norm_train = _param_counts(model.pool_norms)
    head_total, head_train = _param_counts(model.classifier)
    contrast_total, contrast_train = _param_counts(model.contrast_head)
    all_total, all_train = _param_counts(model)

    print(f"backbone:             {back_total:>12,}  (trainable {back_train:,})")
    print(f"probe:                {probe_total:>12,}  (trainable {probe_total:,})")
    print(f"pool LayerNorms:      {norm_total:>12,}  (trainable {norm_train:,})")
    print(f"linear head:          {head_total:>12,}  (trainable {head_train:,})")
    print(f"contrast head:        {contrast_total:>12,}  (trainable {contrast_train:,})")
    print("-" * 44)
    print(f"total params:         {all_total:>12,}")
    print(f"total trainable:      {all_train:>12,}")
    tap_in = model.num_probes * model.hidden_size
    print(f"\nhead in: {tap_in} × {n_taps} taps → {model.classifier.in_features}")
    for block in cfg.lora.blocks:
        if not block.enabled:
            continue
        n = sum(int(p.numel()) for p in lora_params_for_block(model.backbone, block))
        print(f"{block.name} params:       {n:>12,}")

    model.to(device)
    model.eval()
    with torch.no_grad():
        vocab = int(model.backbone.config.vocab_size)
        dummy_ids = torch.randint(0, min(1000, vocab), (2, 32), device=device)
        attn = torch.ones_like(dummy_ids)
        logits = model(dummy_ids, attn)
        print(f"\ndummy forward: logits {tuple(logits.shape)}")
    print()
    return model


if __name__ == "__main__":
    run(load_config())
