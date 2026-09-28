"""Print decision-model architecture, parameter counts, and dummy-forward shapes."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config import CONFIG_PATH, DecisionConfig, apply_hf_cache, load_config
from model.decision_model import ModernBERTDecisionModel
from device import pick_device
from formatting import format_batch, format_candidate_sequence_text


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
        gold = ex.options[ex.answer_index]
        print("model input (gold candidate row):", flush=True)
        print(
            format_candidate_sequence_text(
                state=ex.state,
                question=ex.question,
                options=ex.options,
                candidate=gold,
            ),
            flush=True,
        )
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
        "Sequence:   [CLS] [state] … [question] … [options] … [candidate] … "
        "(one sequence per candidate option)",
        flush=True,
    )
    print(
        "Scoring:    Linear on [CLS] only; train = BCE per candidate row, infer = softmax",
        flush=True,
    )
    print(f"Context:    max {model.max_len} tokens (no truncation)", flush=True)
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
        f"└─ score_head  dropout={cfg.model.score_head_dropout}, one CLS logit per forward",
        flush=True,
    )

    print("\n=== Parameters ===", flush=True)
    enc_b = counts.encoder_trainable * 4
    head_b = counts.score_head * 4
    print(
        f"{'encoder (train):':<22} {_fmt_line(counts.encoder_trainable, enc_b)}  ({enc_label})",
        flush=True,
    )
    print(
        f"{'score_head:':<22} {_fmt_line(counts.score_head, head_b)}  (trainable)",
        flush=True,
    )
    if counts.marker_embeddings:
        print(
            f"{'marker embed:':<22} {_fmt_line(counts.marker_embeddings, counts.marker_embeddings * 4)}",
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
                "state": "Two choices.",
                "question": "Pick one?",
                "options": ["A", "B"],
            },
        ]
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
        print("\n=== Dummy forward ===", flush=True)
        print(f"batch_size:        {b}", flush=True)
        print(f"K_max:             {k}", flush=True)
        print(f"seq_len:           {seq_len}", flush=True)
        print(f"input_ids:         {tuple(batch.input_ids.shape)}", flush=True)
        print(f"attention_mask:    {tuple(batch.attention_mask.shape)}", flush=True)
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
