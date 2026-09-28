"""Hand-written architecture checks (dummy strings/tensors only)."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from checkpoint import save_checkpoint, load_checkpoint
from config import CONFIG_PATH, DecisionConfig, load_config
from device import pick_device
from formatting import build_candidate_sequence, format_batch, format_train_batch, resolve_segment_marker_ids
from model.decision_model import build_model


def _assert_close(name: str, value: float, target: float, tol: float = 1e-4) -> None:
    if abs(value - target) > tol:
        raise AssertionError(f"{name}: expected ~{target}, got {value}")


def run_checks(cfg: DecisionConfig) -> None:
    device = pick_device()
    model = build_model(cfg)
    model.to(device)
    tokenizer = model.tokenizer
    markers = resolve_segment_marker_ids(tokenizer)

    train_rows = [
        ("Alpha state.", "Pick one?", ["A", "B", "C"], "B"),
        ("Beta state.", "Pick one?", ["X", "Y"], "X"),
    ]
    train_batch = format_train_batch(
        train_rows,
        cfg=cfg,
        tokenizer=tokenizer,
        max_len=model.max_len,
        device=device,
    )
    train_labels = torch.tensor([1.0, 1.0], device=device)
    if train_batch.input_ids.dim() != 2:
        raise AssertionError(f"train batch must be [B, L], got {tuple(train_batch.input_ids.shape)}")

    model.train()
    gold_logits = model.score_gold(train_batch.input_ids, train_batch.attention_mask)
    if gold_logits.shape != (2,):
        raise AssertionError(f"gold logits shape {tuple(gold_logits.shape)}")
    train_loss = F.binary_cross_entropy_with_logits(gold_logits, train_labels)
    train_loss.backward()
    if model.score_head.weight.grad is None or model.score_head.weight.grad.abs().sum().item() == 0.0:
        raise AssertionError("score_head should receive gradients from gold BCE")

    model.zero_grad(set_to_none=True)

    requests = [
        {"state": "Alpha state.", "question": "Pick one?", "options": ["A", "B", "C"]},
        {"state": "Beta state.", "question": "Pick one?", "options": ["X", "Y"]},
    ]
    batch = format_batch(
        requests,
        cfg=cfg,
        tokenizer=tokenizer,
        max_len=model.max_len,
        device=device,
    )
    model.eval()
    out = model(batch.input_ids, batch.attention_mask, batch.option_mask)
    if out.logits.shape != (2, 3):
        raise AssertionError(f"logits shape {tuple(out.logits.shape)}")
    if out.probabilities.shape != (2, 3):
        raise AssertionError(f"probabilities shape {tuple(out.probabilities.shape)}")

    row0_sum = float(out.probabilities[0, batch.option_mask[0]].sum().item())
    _assert_close("row0 prob sum", row0_sum, 1.0)
    row1_sum = float(out.probabilities[1, batch.option_mask[1]].sum().item())
    _assert_close("row1 prob sum", row1_sum, 1.0)
    padded = out.probabilities[~batch.option_mask]
    if padded.numel() and float(padded.abs().max().item()) > 0.0:
        raise AssertionError("padded option probabilities must be zero")

    model.train()
    gold_logits = model.score_gold(train_batch.input_ids, train_batch.attention_mask)
    loss = gold_logits.sum()
    loss.backward()
    embed = model.encoder.get_input_embeddings()
    if embed is not None and cfg.train.train_cls_embedding:
        if model.cls_token_id is None:
            raise AssertionError("cls_token_id must be set when train_cls_embedding is true")
        emb_grad = embed.weight.grad
        if emb_grad is None:
            raise AssertionError("CLS embedding row should receive gradients")
        cls_id = int(model.cls_token_id)
        if float(emb_grad[cls_id].abs().sum().item()) == 0.0:
            raise AssertionError("CLS embedding row should receive non-zero gradients")
        mask = torch.ones(emb_grad.shape[0], dtype=torch.bool)
        mask[cls_id] = False
        if float(emb_grad[mask].abs().sum().item()) != 0.0:
            raise AssertionError("non-CLS embedding rows should not receive gradients")
    n_ft = model.finetune_last_n_layers
    layers = model.encoder.layers
    start = len(layers) - n_ft if n_ft else len(layers)
    train_mask = model.encoder_layer_train
    for layer_idx, layer in enumerate(layers):
        layer_has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0.0 for p in layer.parameters()
        )
        if layer_idx >= start and n_ft > 0:
            slot = layer_idx - start
            should_train = train_mask[slot]
            if should_train and not layer_has_grad:
                raise AssertionError(f"trainable encoder layer {layer_idx} should receive gradients")
            if not should_train and layer_has_grad:
                raise AssertionError(f"disabled encoder layer {layer_idx} should not receive gradients")
        elif layer_has_grad:
            raise AssertionError(f"frozen encoder layer {layer_idx} should not receive gradients")
    if n_ft and any(train_mask):
        norm_has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0.0
            for p in model.encoder.final_norm.parameters()
        )
        if not norm_has_grad:
            raise AssertionError("final_norm should receive gradients when any encoder block is trainable")
    elif n_ft and not any(train_mask):
        norm_has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0.0
            for p in model.encoder.final_norm.parameters()
        )
        if norm_has_grad:
            raise AssertionError("final_norm should stay frozen when all encoder blocks are off")
    model.zero_grad(set_to_none=True)

    huge = "token " * (model.max_len * 4)
    try:
        build_candidate_sequence(
            tokenizer=tokenizer,
            markers=markers,
            state=huge,
            question="Q?",
            options=["ok"],
            candidate="ok",
            max_len=model.max_len,
        )
        raise AssertionError("oversized input should be rejected")
    except ValueError:
        pass

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "ckpt"
        model.eval()
        save_checkpoint(model, root, at_end=True)
        reloaded = load_checkpoint(root, cfg)
        reloaded.to(device)
        reloaded.eval()
        with torch.inference_mode():
            before = model(batch.input_ids, batch.attention_mask, batch.option_mask)
            after = reloaded(batch.input_ids, batch.attention_mask, batch.option_mask)
        if not torch.allclose(before.logits, after.logits, atol=1e-5, rtol=1e-5):
            raise AssertionError("forward logits differ after checkpoint reload")

    print("verify: all checks passed", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run architecture checks for ModernBERT decision model")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args(argv)
    run_checks(load_config(args.config))


if __name__ == "__main__":
    main()
