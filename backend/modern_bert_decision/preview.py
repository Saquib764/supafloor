"""Print decision-model architecture, parameter counts, and dummy-forward shapes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from config import CONFIG_PATH, DecisionConfig, apply_hf_cache, load_config
from model.decision_model import ModernBERTDecisionModel
from device import pick_device
from formatting import (
    build_candidate_sequence,
    format_batch,
    format_candidate_segment,
    resolve_sequence_boundary_ids,
)


def _fmt_params(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    if n >= 1_000:
        return f"{n / 1e3:.2f}K"
    return str(n)


def _fmt_mb(num_bytes: int) -> str:
    return f"{num_bytes / (1024 ** 2):.2f} MB"


def _fmt_line(n: int, num_bytes: int) -> str:
    return f"{n:,} ({_fmt_params(n)}, {_fmt_mb(num_bytes)})"


def _default_preview_requests(_cfg: DecisionConfig) -> list[dict[str, object]]:
    path = CONFIG_PATH.parent / "example_request.json"
    if path.is_file():
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "options" in raw:
            return [raw]
    return [
        {
            "state": "Sample passage for preview.",
            "question": "Which answer is correct?",
            "options": ["First choice", "Second choice", "Third choice", "Fourth choice"],
        },
    ]


def _clip(text: str, limit: int = 280) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _print_training_samples(
    cfg: DecisionConfig,
    *,
    tokenizer,
    max_len: int,
    count: int = 2,
) -> list[dict[str, object]]:
    from data import example_to_request, load_race_preview_examples

    print("\n=== Training data sample ===", flush=True)
    print(f"dataset:      {cfg.data.dataset}", flush=True)
    print(f"source:       {cfg.data.source}", flush=True)
    print(f"train_split:  {cfg.data.train_split}", flush=True)
    print(f"race_subset:  {cfg.data.race_subset}", flush=True)
    try:
        examples = load_race_preview_examples(cfg, count=count)
    except Exception as exc:
        print(f"(could not load samples: {exc})", flush=True)
        return []
    if not examples:
        print("(no rows)", flush=True)
        return []
    boundaries = resolve_sequence_boundary_ids(tokenizer)
    requests: list[dict[str, object]] = []
    total_tokens = 0
    for index, ex in enumerate(examples, start=1):
        requests.append(example_to_request(ex))
        print(f"\n--- example {index} ---", flush=True)
        print(f"state:         {_clip(ex.state)}", flush=True)
        print(f"question:      {ex.question}", flush=True)
        print(f"answer_index:  {ex.answer_index}", flush=True)
        n_opts = len(ex.options)
        print(f"  options: {n_opts} (one encoder forward + logit each; softmax over all)", flush=True)
        token_lens: list[int] = []
        for opt_index, option in enumerate(ex.options):
            mark = " *" if opt_index == ex.answer_index else ""
            seq = build_candidate_sequence(
                tokenizer=tokenizer,
                boundaries=boundaries,
                state=ex.state,
                question=ex.question,
                candidate=option,
                max_len=max_len,
            )
            token_lens.append(len(seq))
            total_tokens += len(seq)
            print(
                f"  [{opt_index}] tokens={len(seq)}{mark}  {_clip(option, 120)}",
                flush=True,
            )
        if token_lens and len(set(token_lens)) == 1:
            print(
                f"  (each option sequence length {token_lens[0]} — state+question shared; "
                "candidate segment differs)",
                flush=True,
            )
        gold = ex.options[ex.answer_index]
        print("  segment sketch (full state in model input; not repeated here):", flush=True)
        print(f"    [CLS] <state passage>", flush=True)
        print(f"    [SEP] {ex.question.strip()}", flush=True)
        print(f"    [SEP] {format_candidate_segment(gold)}", flush=True)
    print(f"\ntotal tokens (all option sequences, {len(examples)} examples): {total_tokens}", flush=True)
    return requests


def print_model_summary(
    cfg: DecisionConfig,
    *,
    dummy_forward: bool = True,
    show_samples: bool = True,
    sample_count: int = 2,
) -> ModernBERTDecisionModel:
    apply_hf_cache(cfg.hf_cache)
    from model.decision_model import build_model

    model = build_model(cfg)
    from model.params import parameter_breakdown

    counts = parameter_breakdown(model)

    print("\n=== Architecture ===", flush=True)
    print(f"Backbone:   {cfg.pretrained_name} (encoder only, no MLM head)", flush=True)
    print(
        "Sequence:   [CLS] state … [SEP] question … [SEP] candidate prompt … "
        "(one sequence per option; candidate = template + option text)",
        flush=True,
    )
    print(
        "Scoring:    Linear on [CLS]; softmax over every option in the request (train/eval/infer)",
        flush=True,
    )
    print(f"Context:    max {model.max_len} tokens (no truncation)", flush=True)
    lora = cfg.train.lora
    if model.lora_enabled:
        from model.lora import resolve_lora_layer_indices

        n_ft = model.finetune_last_n_layers
        n_layers = len(model.encoder.layers)
        lora_blocks = resolve_lora_layer_indices(
            n_layers,
            lora,
            skip_last_n_layers=n_ft,
        )
        blocks_label = ",".join(str(i) for i in lora_blocks)
        enc_label = f"LoRA rank {lora.rank}, alpha {lora.alpha} (blocks [{blocks_label}] / {n_layers})"
        print(f"Encoder:    {enc_label}", flush=True)
        print(
            f"            targets: {', '.join(lora.target_modules)} "
            f"(not score_head)",
            flush=True,
        )
        print(f"            lr:      {lora.learning_rate}", flush=True)
    else:
        n_ft = model.finetune_last_n_layers
        if n_ft:
            on = sum(model.encoder_layer_train)
            enc_label = f"last {n_ft} block(s), {on} trainable"
        else:
            enc_label = "frozen"
        print(f"Encoder:    {enc_label} (finetune_last_n_layers={n_ft})", flush=True)
        if n_ft:
            print(f"            train: {list(model.encoder_layer_train)}", flush=True)
            print(f"            LRs:   {list(model.encoder_finetune_lrs)}", flush=True)
    print(flush=True)
    print("ModernBERTDecisionModel", flush=True)
    print(f"├─ encoder   pretrained ModernBERT blocks", flush=True)
    print(
        f"└─ score_head  dropout={cfg.model.score_head_dropout}, lr={cfg.train.head_learning_rate}, "
        "one CLS logit per forward",
        flush=True,
    )

    print("\n=== Parameters ===", flush=True)
    enc_b = counts.encoder_trainable * 4
    head_b = counts.score_head * 4
    if model.lora_enabled:
        lora_b = counts.lora * 4
        print(
            f"{'lora:':<22} {_fmt_line(counts.lora, lora_b)}  ({enc_label})",
            flush=True,
        )
    else:
        print(
            f"{'encoder (train):':<22} {_fmt_line(counts.encoder_trainable, enc_b)}  ({enc_label})",
            flush=True,
        )
    print(
        f"{'score_head:':<22} {_fmt_line(counts.score_head, head_b)}  (trainable)",
        flush=True,
    )
    if counts.cls_embedding:
        print(
            f"{'CLS embedding:':<22} {_fmt_line(counts.cls_embedding, counts.cls_embedding * 4)}  (trainable row)",
            flush=True,
        )
    print(f"{'frozen:':<22} {_fmt_line(counts.frozen, counts.frozen * 4)}", flush=True)
    print(f"{'trainable total:':<22} {_fmt_line(counts.trainable, counts.trainable * 4)}", flush=True)
    print(f"{'total:':<22} {_fmt_line(counts.total, counts.total * 4)}", flush=True)

    sample_requests: list[dict[str, object]] = []
    if show_samples:
        sample_requests = _print_training_samples(
            cfg,
            tokenizer=model.tokenizer,
            max_len=model.max_len,
            count=sample_count,
        )

    if dummy_forward:
        device = pick_device()
        model.to(device)
        requests = sample_requests if sample_requests else _default_preview_requests(cfg)
        batch = format_batch(
            requests,
            cfg=cfg,
            tokenizer=model.tokenizer,
            max_len=model.max_len,
            device=device,
        )
        with torch.inference_mode():
            out = model(
                batch.input_ids,
                batch.attention_mask,
                batch.option_mask,
            )
        b, k, seq_len = batch.input_ids.shape
        print("\n=== Dummy forward (softmax over all options per row) ===", flush=True)
        print(f"batch_size:        {b}", flush=True)
        print(f"K_max:             {k}", flush=True)
        print(f"seq_len:           {seq_len}", flush=True)
        print(f"input_ids:         {tuple(batch.input_ids.shape)}", flush=True)
        print(f"attention_mask:    {tuple(batch.attention_mask.shape)}", flush=True)
        print(f"option_mask:       {tuple(batch.option_mask.shape)}", flush=True)
        print(f"logits:            {tuple(out.logits.shape)}", flush=True)
        print(f"probabilities:     {tuple(out.probabilities.shape)}", flush=True)
        print(f"selected_indices:  {tuple(out.selected_indices.shape)}", flush=True)
        for row, req in enumerate(requests):
            raw_opts = req.get("options")
            if not isinstance(raw_opts, list):
                continue
            options = [str(o) for o in raw_opts]
            n_opts = len(options)
            mask = batch.option_mask[row].cpu()
            if int(mask.sum().item()) != n_opts:
                raise RuntimeError(
                    f"preview row {row}: option_mask count {int(mask.sum())} != {n_opts} options"
                )
            print(f"\n  row {row + 1}: {n_opts} options scored", flush=True)
            for i, opt in enumerate(options):
                prob = float(out.probabilities[row, i].item())
                logit = float(out.logits[row, i].item())
                print(
                    f"    [{i}] p={prob:.4f} logit={logit:.4f}  {_clip(opt, 100)}",
                    flush=True,
                )
            sel = int(out.selected_indices[row].item())
            print(f"    argmax → [{sel}] {_clip(options[sel], 100)}", flush=True)
        print(
            "\nNote: untrained scoring head — softmax values are not meaningful decisions.",
            flush=True,
        )

    print("\n=== Config ===", flush=True)
    print(f"pretrained_name: {cfg.pretrained_name}", flush=True)
    print(f"hf_cache:        {cfg.hf_cache}", flush=True)
    print(f"infer_ckpt:      {cfg.infer.checkpoint}", flush=True)
    print(flush=True)
    return model


def run(cfg: DecisionConfig, *, show_samples: bool = True, sample_count: int = 2) -> None:
    print_model_summary(cfg, show_samples=show_samples, sample_count=sample_count)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Print ModernBERT decision model summary")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--no-samples",
        action="store_true",
        help="Skip loading RACE preview rows from the configured train split",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=5,
        help="Number of training rows to print (default: 5)",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.no_samples:
        print_model_summary(cfg, show_samples=False)
    else:
        print_model_summary(cfg, show_samples=True, sample_count=max(1, args.sample_count))


if __name__ == "__main__":
    main()
