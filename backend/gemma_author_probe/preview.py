"""Print model summary: frozen Gemma, probe token, multi-layer head."""

from __future__ import annotations

import torch

from config import Config, load_config
from device import pick_device
from labels import MODEL_NAMES, NUM_CLASSES
from model import ProbeClassifier, build_model


def _param_counts(module: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def _describe_head(cfg: Config) -> list[str]:
    return [
        "type:           LayerNorm per block, concat, dropout, Linear",
        f"tap_layers:     {list(cfg.tap_layers)}",
        f"dropout:        {cfg.train.head_dropout:g}",
        f"num_classes:    {NUM_CLASSES}",
        f"probe tokens:   {cfg.num_probes} (fixed order, concat hiddens per tap)",
        "forward:        all probe hiddens at each tap → LN → concat taps → dropout → Linear",
        "probe init:     each probe = mean token embed of one structure question (low→semantic)",
        "loss:           cross entropy (+ optional contrastive on contrast_head)",
        f"contrast_head:  Linear→ReLU→Linear→L2 "
        f"({cfg.train.contrast_head.hidden_dim}→{cfg.train.contrast_head.out_dim})",
    ]


def run(cfg: Config | None = None, *, build_from_hub: bool = True) -> ProbeClassifier | None:
    cfg = cfg or load_config()
    device = pick_device()
    train = cfg.train
    n_taps = len(cfg.tap_layers)

    print("\n=== Name That LLM (Gemma blocks + probe + 18-class head) ===\n")
    print(f"model_id:       {cfg.model_id}")
    print(
        f"max_len:        {cfg.max_len}  "
        f"(tokenize to {cfg.max_len - cfg.num_probes}, append {cfg.num_probes} probes)"
    )
    print(f"num_probes:     {cfg.num_probes}")
    print(f"tap_layers:     {list(cfg.tap_layers)}  (1-indexed blocks)")
    print(
        f"input:          passage + {cfg.num_probes} probes; "
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
    print(f"freeze_blocks:    {train.freeze_blocks}")
    print(f"freeze_probes:    {train.freeze_probes}")
    print(f"freeze_head:      {train.freeze_head}")
    print(f"freeze_contrast:  {train.contrast_head.freeze}")
    print(f"embeddings:       {'frozen' if train.freeze_embeddings else 'trainable'}")
    if not train.freeze_blocks:
        print(f"block lr:         {train.block_lr:g}")
    else:
        print("block lr:         n/a (blocks frozen)")
    if not train.freeze_probes:
        print(f"probe lr:         {train.probe_lr:g}")
    else:
        print("probe lr:         n/a (probes frozen)")
    if not train.freeze_head:
        print(f"head lr:          {train.head_lr:g}  (pool norms + linear)")
    else:
        print("head lr:          n/a (head frozen)")
    if not train.contrast_head.freeze:
        print(f"contrast_head lr: {train.contrast_head.lr:g}")
    else:
        print("contrast_head lr: n/a (contrast head frozen)")
    print(
        f"loss:             cross entropy "
        f"(batch={train.batch_size}: {train.prompt_count} prompts × "
        f"{train.batch_size // train.prompt_count} models, "
        f"micro={'off' if not train.micro_batch else train.micro_batch_size})"
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
