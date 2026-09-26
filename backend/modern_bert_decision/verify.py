"""Hand-written architecture checks (dummy strings/tensors only)."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch

from checkpoint import save_checkpoint, load_checkpoint
from config import CONFIG_PATH, DecisionConfig, load_config
from device import pick_device
from formatting import build_one_sequence, format_batch
from model.decision_model import build_model


def _assert_close(name: str, value: float, target: float, tol: float = 1e-4) -> None:
    if abs(value - target) > tol:
        raise AssertionError(f"{name}: expected ~{target}, got {value}")


def run_checks(cfg: DecisionConfig) -> None:
    device = pick_device()
    model = build_model(cfg)
    model.to(device)
    markers = model.markers
    tokenizer = model.tokenizer
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    assert cls_id is not None and sep_id is not None

    requests = [
        {
            "state": "Alpha state.",
            "question": "Pick one?",
            "options": ["A", "B", "C"],
        },
        {
            "state": "Beta state.",
            "question": "Pick one?",
            "options": ["X"],
        },
    ]
    batch = format_batch(
        requests,
        tokenizer=tokenizer,
        markers=markers,
        max_len=model.max_len,
        device=device,
    )
    if batch.option_mask.shape != (2, 3):
        raise AssertionError(f"expected option_mask (2, 3), got {tuple(batch.option_mask.shape)}")
    if not batch.option_mask[1, 0].item() or batch.option_mask[1, 1:].any().item():
        raise AssertionError("mixed batch option_mask row incorrect")

    model.train()
    out = model(
        batch.input_ids,
        batch.attention_mask,
        batch.option_positions,
        batch.option_mask,
    )
    if out.logits.shape != (2, 3):
        raise AssertionError(f"logits shape {tuple(out.logits.shape)}")
    if out.probabilities.shape != (2, 3):
        raise AssertionError(f"probabilities shape {tuple(out.probabilities.shape)}")
    if out.selected_indices.shape != (2,):
        raise AssertionError(f"selected_indices shape {tuple(out.selected_indices.shape)}")

    row0_sum = float(out.probabilities[0, batch.option_mask[0]].sum().item())
    _assert_close("row0 prob sum", row0_sum, 1.0)
    row1_sum = float(out.probabilities[1, batch.option_mask[1]].sum().item())
    _assert_close("row1 prob sum", row1_sum, 1.0)
    if float(out.probabilities[1, 0].item()) != 1.0:
        raise AssertionError("single-option row should have probability 1 on the only option")
    padded = out.probabilities[~batch.option_mask]
    if padded.numel() and float(padded.abs().max().item()) > 0.0:
        raise AssertionError("padded option probabilities must be zero")

    loss = out.logits[batch.option_mask].sum()
    loss.backward()
    if model.score_head.weight.grad is None or model.score_head.weight.grad.abs().sum().item() == 0.0:
        raise AssertionError("score_head should receive gradients")
    embed = model.encoder.get_input_embeddings()
    if embed is None:
        raise AssertionError("encoder must expose input embeddings")
    emb_grad = embed.weight.grad
    if emb_grad is None:
        raise AssertionError("marker embedding rows should receive gradients")
    base = model.base_vocab_size
    if not cfg.model.train_pretrained_encoder:
        if float(emb_grad[:base].abs().sum().item()) != 0.0:
            raise AssertionError("pretrained embedding rows should not receive gradients")
        if float(emb_grad[base:].abs().sum().item()) == 0.0:
            raise AssertionError("marker embedding rows should receive gradients")
        frozen_layer_grad = False
        for name, param in model.encoder.named_parameters():
            if "embed" in name.lower():
                continue
            if param.requires_grad:
                continue
            if param.grad is not None and param.grad.abs().sum().item() > 0.0:
                frozen_layer_grad = True
                break
        if frozen_layer_grad:
            raise AssertionError("frozen encoder layers should not receive gradients")
    else:
        enc_has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0.0
            for p in model.encoder.parameters()
            if p.requires_grad
        )
        if not enc_has_grad:
            raise AssertionError("encoder should receive gradients when not frozen")
    model.zero_grad(set_to_none=True)

    huge = "token " * (model.max_len * 4)
    try:
        build_one_sequence(
            tokenizer=tokenizer,
            markers=markers,
            cls_id=cls_id,
            sep_id=sep_id,
            state=huge,
            question="Q?",
            options=["ok"],
            max_len=model.max_len,
        )
        raise AssertionError("oversized input should be rejected")
    except ValueError:
        pass

    try:
        format_batch(
            [{"state": "s", "question": "q", "options": []}],
            tokenizer=tokenizer,
            markers=markers,
            max_len=model.max_len,
        )
        raise AssertionError("empty options should be rejected")
    except ValueError:
        pass

    model.eval()
    with torch.inference_mode():
        before = model(
            batch.input_ids,
            batch.attention_mask,
            batch.option_positions,
            batch.option_mask,
        )
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "ckpt"
        save_checkpoint(model, root, at_end=True)
        reloaded = load_checkpoint(root, cfg)
        reloaded.to(device)
        reloaded.eval()
        if not torch.equal(model.score_head.weight.cpu(), reloaded.score_head.weight.cpu()):
            raise AssertionError("score_head weights differ after reload")
        with torch.inference_mode():
            after = reloaded(
                batch.input_ids,
                batch.attention_mask,
                batch.option_positions,
                batch.option_mask,
            )
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
