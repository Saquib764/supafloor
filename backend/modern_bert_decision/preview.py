"""Print decision-model architecture, parameter counts, and dummy-forward shapes."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config import CONFIG_PATH, DecisionConfig, apply_hf_cache, load_config
from model.decision_model import ModernBERTDecisionModel
from device import pick_device
from formatting import format_batch


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


def _clip(text: str, limit: int = 280) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _print_training_samples(cfg: DecisionConfig, count: int = 2) -> list[dict[str, object]]:
    from data import example_to_request, load_race_preview_examples

    print("\n=== Training data sample (with augment) ===", flush=True)
    print(f"dataset:      {cfg.data.dataset}", flush=True)
    print(f"source:       {cfg.data.source}", flush=True)
    print(f"train_split:  {cfg.data.train_split}", flush=True)
    print(f"race_subset:  {cfg.data.race_subset}", flush=True)
    print(f"option_subset_prob: {cfg.data.option_subset_prob}", flush=True)
    print(f"none_of_these:    {cfg.data.none_of_these_option!r}", flush=True)
    print("  (random subset keep ≥2, optional None-of-these, then shuffle)", flush=True)
    try:
        examples = load_race_preview_examples(cfg, count=count, augment=True)
    except Exception as exc:
        print(f"(could not load samples: {exc})", flush=True)
        return []
    if not examples:
        print("(no rows)", flush=True)
        return []
    requests: list[dict[str, object]] = []
    for index, ex in enumerate(examples, start=1):
        requests.append(example_to_request(ex))
        print(f"\n--- example {index} ---", flush=True)
        print(f"state:         {_clip(ex.state)}", flush=True)
        print(f"question:      {ex.question}", flush=True)
        for opt_index, option in enumerate(ex.options):
            mark = " *" if opt_index == ex.answer_index else ""
            print(f"  option[{opt_index}]: {_clip(option, 120)}{mark}", flush=True)
        print(f"answer_index:  {ex.answer_index}", flush=True)
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
    print(f"Markers:    {cfg.marker_tokens.state}, {cfg.marker_tokens.question}, {cfg.marker_tokens.option}", flush=True)
    print(f"Sequence:   [CLS] [STATE] … [QUESTION] … [OPTION] option_i … [SEP]", flush=True)
    print(f"Scoring:    shared Linear({model.hidden_size}, 1, bias=False) on each [OPTION] hidden", flush=True)
    print(f"Context:    max {model.max_len} tokens (no truncation)", flush=True)
    enc_label = "trainable" if model.train_pretrained_encoder else "frozen"
    print(
        f"Encoder:    {enc_label} "
        f"(config model.train_pretrained_encoder={model.train_pretrained_encoder})",
        flush=True,
    )
    print(flush=True)
    print("ModernBERTDecisionModel", flush=True)
    print(f"├─ encoder   pretrained ModernBERT blocks (+ 3 marker embedding rows)", flush=True)
    print(f"└─ score_head  one logit per option column", flush=True)

    print("\n=== Parameters (original vs new) ===", flush=True)
    orig_b = counts.original_pretrained * 4
    marker_b = counts.marker_embeddings * 4
    head_b = counts.score_head * 4
    print(
        f"{'pretrained encoder:':<22} {_fmt_line(counts.original_pretrained, orig_b)}  ({enc_label})",
        flush=True,
    )
    print(
        f"{'marker embeddings:':<22} {_fmt_line(counts.marker_embeddings, marker_b)}  (3 × {model.hidden_size}, trainable)",
        flush=True,
    )
    print(
        f"{'score_head:':<22} {_fmt_line(counts.score_head, head_b)}  (trainable)",
        flush=True,
    )
    print(f"{'new (trainable):':<22} {_fmt_line(counts.new_trainable, counts.new_trainable * 4)}", flush=True)
    print(f"{'frozen:':<22} {_fmt_line(counts.frozen, counts.frozen * 4)}", flush=True)
    print(f"{'trainable total:':<22} {_fmt_line(counts.trainable, counts.trainable * 4)}", flush=True)
    print(f"{'total:':<22} {_fmt_line(counts.total, counts.total * 4)}", flush=True)
    if not model.train_pretrained_encoder:
        print(
            "  (pretrained embedding rows are grad-masked; only marker rows update)",
            flush=True,
        )

    sample_requests: list[dict[str, object]] = []
    if show_samples:
        sample_requests = _print_training_samples(cfg, count=sample_count)

    if dummy_forward:
        device = pick_device()
        model.to(device)
        requests = sample_requests if sample_requests else [
            {
                "state": "Room A is north of the hallway.",
                "question": "Which door connects to the kitchen?",
                "options": ["Door 1", "Door 2"],
            },
            {
                "state": "Single option example.",
                "question": "Pick the only label.",
                "options": ["Only choice"],
            },
        ]
        batch = format_batch(
            requests,
            tokenizer=model.tokenizer,
            markers=model.markers,
            max_len=model.max_len,
            device=device,
        )
        with torch.inference_mode():
            out = model(
                batch.input_ids,
                batch.attention_mask,
                batch.option_positions,
                batch.option_mask,
            )
        b, k = batch.option_mask.shape
        print("\n=== Dummy forward ===", flush=True)
        print(f"batch_size:        {b}", flush=True)
        print(f"K_max:             {k}", flush=True)
        print(f"input_ids:         {tuple(batch.input_ids.shape)}", flush=True)
        print(f"attention_mask:    {tuple(batch.attention_mask.shape)}", flush=True)
        print(f"option_positions:  {tuple(batch.option_positions.shape)}", flush=True)
        print(f"option_mask:       {tuple(batch.option_mask.shape)}", flush=True)
        print(f"logits:            {tuple(out.logits.shape)}", flush=True)
        print(f"probabilities:     {tuple(out.probabilities.shape)}", flush=True)
        print(f"selected_indices:  {tuple(out.selected_indices.shape)}", flush=True)
        print(
            "\nNote: untrained scoring head — softmax values are not meaningful decisions.",
            flush=True,
        )

    print("\n=== Config ===", flush=True)
    print(f"pretrained_name: {cfg.pretrained_name}", flush=True)
    print(f"hf_cache:        {cfg.hf_cache}", flush=True)
    print(f"train_encoder:   {cfg.model.train_pretrained_encoder}", flush=True)
    t = cfg.train
    print(f"session_dir:     {t.session_dir}", flush=True)
    print(f"epochs:          {t.epochs}", flush=True)
    print(f"batch_size:      {t.batch_size}", flush=True)
    print(f"grad_accum:      {t.grad_accum}", flush=True)
    print(f"learning_rate:   {t.learning_rate}", flush=True)
    print(f"init_checkpoint: {t.init_checkpoint or '(none)'}", flush=True)
    print(f"resume:          {t.resume}", flush=True)
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
