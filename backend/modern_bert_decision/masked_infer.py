"""Vanilla ModernBERT MLM: one sequence per question, [MASK] per option."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, PreTrainedTokenizerBase

from config import (
    CONFIG_PATH,
    SCORE_TOKEN_MASK,
    DecisionConfig,
    assert_hub_has_space,
    hub_from_pretrained_kwargs,
    load_config,
)
from data import DecisionExample, pick_random_examples
from device import pick_device
from formatting import (
    cls_positions,
    format_masked_batch,
    format_masked_input_text,
    max_sequence_length,
)


@dataclass(frozen=True)
class MaskTopPrediction:
    token: str
    logit: float


@dataclass(frozen=True)
class MaskedOptionPrediction:
    option: str
    mask_index: int
    top_token: str
    top_logit: float
    true_logit: float
    false_logit: float
    true_probability: float
    top_predictions: tuple[MaskTopPrediction, ...]
    cls_index: int
    cls_top_token: str
    cls_top_logit: float
    cls_top_predictions: tuple[MaskTopPrediction, ...]


@dataclass(frozen=True)
class MaskedPrediction:
    selected_index: int
    input_text: str
    options: list[MaskedOptionPrediction]


def _single_token_id(tokenizer: PreTrainedTokenizerBase, word: str) -> int:
    ids = tokenizer.encode(word, add_special_tokens=False)
    if len(ids) != 1:
        tokens = tokenizer.convert_ids_to_tokens(ids)
        raise ValueError(
            f"label {word!r} must be a single tokenizer token, got ids={ids} tokens={tokens}"
        )
    return int(ids[0])


def _mlm_top_predictions(
    row: torch.Tensor,
    tokenizer: PreTrainedTokenizerBase,
    *,
    top_k: int,
) -> tuple[str, float, tuple[MaskTopPrediction, ...]]:
    k = min(top_k, row.numel())
    topk = torch.topk(row, k=k)
    top_preds: list[MaskTopPrediction] = []
    for idx, logit in zip(topk.indices.tolist(), topk.values.tolist(), strict=True):
        top_preds.append(
            MaskTopPrediction(
                token=str(tokenizer.convert_ids_to_tokens(int(idx))),
                logit=float(logit),
            )
        )
    top_idx = int(topk.indices[0].item())
    return (
        str(tokenizer.convert_ids_to_tokens(top_idx)),
        float(row[top_idx].item()),
        tuple(top_preds),
    )


def load_mlm(cfg: DecisionConfig, device: torch.device):
    hub_kw = hub_from_pretrained_kwargs(cfg)
    assert_hub_has_space(cfg)
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_name, use_fast=True, **hub_kw)
    except OSError as exc:
        raise RuntimeError(f"failed to load tokenizer {cfg.pretrained_name!r}") from exc
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.sep_token is not None:
            tokenizer.pad_token = tokenizer.sep_token
    try:
        model = AutoModelForMaskedLM.from_pretrained(cfg.pretrained_name, **hub_kw)
    except OSError as exc:
        raise RuntimeError(f"failed to load MLM {cfg.pretrained_name!r}") from exc
    max_len = max_sequence_length(tokenizer, model.config)
    model.to(device)
    model.eval()
    return model, tokenizer, max_len


@torch.inference_mode()
def predict_masked(
    model,
    tokenizer: PreTrainedTokenizerBase,
    *,
    state: str,
    question: str,
    options: list[str],
    max_len: int,
    device: torch.device,
    top_k: int = 5,
) -> MaskedPrediction:
    cls_id = tokenizer.cls_token_id
    if cls_id is None:
        raise ValueError("tokenizer must define cls_token_id")
    cls_id = int(cls_id)
    true_id = _single_token_id(tokenizer, "true")
    false_id = _single_token_id(tokenizer, "false")

    request = {"state": state, "question": question, "options": options}
    batch = format_masked_batch(
        [request],
        tokenizer=tokenizer,
        max_len=max_len,
        score_token=SCORE_TOKEN_MASK,
        device=device,
    )
    seq_len = int(batch.attention_mask[0].sum().item())
    seq = batch.input_ids[0, :seq_len].tolist()
    n_opts = int(batch.option_mask[0].sum().item())
    positions = [int(batch.score_index[0, j].item()) for j in range(n_opts)]

    cls_pos = cls_positions(seq, cls_id)
    if len(positions) != len(options):
        raise RuntimeError(
            f"expected {len(options)} [MASK] positions, found {len(positions)} in sequence"
        )
    if len(cls_pos) != len(options):
        raise RuntimeError(
            f"expected {len(options)} [CLS] positions, found {len(cls_pos)} in sequence"
        )

    sep_display = tokenizer.sep_token or "[SEP]"
    mask_display = tokenizer.mask_token or "[MASK]"
    cls_display = tokenizer.cls_token or "[CLS]"
    layout_text = format_masked_input_text(
        state=state,
        question=question,
        options=options,
        mask_token=mask_display,
        sep_token=sep_display,
        cls_token=cls_display,
    )
    decoded_text = tokenizer.decode(seq, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    input_text = f"{layout_text}\n\n--- tokenized ({len(seq)} tokens) ---\n{decoded_text}"

    input_ids = batch.input_ids[:, :seq_len]
    attention_mask = batch.attention_mask[:, :seq_len]
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0]

    option_preds: list[MaskedOptionPrediction] = []
    true_probs: list[float] = []
    for opt, pos, cls_i in zip(options, positions, cls_pos, strict=True):
        row = logits[pos]
        top_token, top_logit, top_preds = _mlm_top_predictions(row, tokenizer, top_k=top_k)
        true_logit = float(row[true_id].item())
        false_logit = float(row[false_id].item())
        pair = torch.tensor([true_logit, false_logit], device=row.device)
        true_prob = float(torch.softmax(pair, dim=0)[0].item())
        true_probs.append(true_prob)

        cls_row = logits[cls_i]
        cls_top_token, cls_top_logit, cls_top_preds = _mlm_top_predictions(
            cls_row, tokenizer, top_k=top_k
        )

        option_preds.append(
            MaskedOptionPrediction(
                option=opt,
                mask_index=pos,
                top_token=top_token,
                top_logit=top_logit,
                true_logit=true_logit,
                false_logit=false_logit,
                true_probability=true_prob,
                top_predictions=top_preds,
                cls_index=cls_i,
                cls_top_token=cls_top_token,
                cls_top_logit=cls_top_logit,
                cls_top_predictions=cls_top_preds,
            )
        )

    selected = int(max(range(len(true_probs)), key=true_probs.__getitem__))
    return MaskedPrediction(selected_index=selected, input_text=input_text, options=option_preds)


def run_masked_infer(
    cfg: DecisionConfig,
    *,
    samples: int,
    split: str | None = None,
    seed: int | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
) -> tuple[list[MaskedPrediction], list[DecisionExample]]:
    device = device or pick_device()
    split_name = split or cfg.data.val_split
    seed_val = cfg.data.seed if seed is None else seed
    examples = pick_random_examples(cfg, split=split_name, count=samples, seed=seed_val)
    if not examples:
        raise RuntimeError(
            f"no examples from dataset {cfg.data.dataset!r} split {split_name!r}; "
            "run download-data or check data.source"
        )
    if len(examples) < samples:
        print(
            f"masked-infer: only {len(examples)} rows in split {split_name!r} "
            f"(requested {samples})",
            flush=True,
        )

    print(
        f"masked-infer: {cfg.pretrained_name} on {len(examples)} random rows "
        f"({cfg.data.dataset}/{split_name}, seed={seed_val})",
        flush=True,
    )
    model, tokenizer, max_len = load_mlm(cfg, device)
    cls_display = tokenizer.cls_token or "[CLS]"

    predictions: list[MaskedPrediction] = []
    correct = 0
    for i, ex in enumerate(examples):
        pred = predict_masked(
            model,
            tokenizer,
            state=ex.state,
            question=ex.question,
            options=ex.options,
            max_len=max_len,
            device=device,
        )
        predictions.append(pred)
        hit = pred.selected_index == ex.answer_index
        if hit:
            correct += 1
        if verbose:
            print(f"\n--- sample {i + 1}/{len(examples)} ---")
            print("=== Input ===")
            print(pred.input_text, flush=True)
            print("\n=== Mask outputs ===")
            for j, row in enumerate(pred.options):
                mark = " *" if j == pred.selected_index else ""
                gold = " (gold)" if j == ex.answer_index else ""
                print(f"\n[{j}]{mark}{gold} mask position {row.mask_index}")
                print(f"  option: {row.option}")
                for rank, tp in enumerate(row.top_predictions, start=1):
                    print(f"  mask top-{rank}: {tp.token!r}  logit={tp.logit:.4f}")
                print(
                    f"  mask true/false: logit(true)={row.true_logit:.4f} "
                    f"logit(false)={row.false_logit:.4f} P(true)={row.true_probability:.4f}"
                )
                print(f"  CLS position {row.cls_index} (token {cls_display!r})")
                print(f"  CLS MLM top-1: {row.cls_top_token!r}  logit={row.cls_top_logit:.4f}")
                for rank, tp in enumerate(row.cls_top_predictions, start=1):
                    print(f"  CLS top-{rank}: {tp.token!r}  logit={tp.logit:.4f}")
            print(
                json.dumps(
                    {
                        "selected_index": pred.selected_index,
                        "answer_index": ex.answer_index,
                        "correct": hit,
                    },
                    indent=2,
                )
            )

    acc = correct / len(examples) if examples else 0.0
    print(
        f"\nmasked-infer accuracy: {correct}/{len(examples)} ({acc:.1%}) "
        f"on {cfg.data.dataset}/{split_name}",
        flush=True,
    )
    return predictions, examples


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="ModernBERT MLM inference: [MASK] per option, score via P(true)"
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--samples",
        type=int,
        default=2,
        help="Number of random dataset rows (default: 20)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split (default: data.val_split from config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for row selection (default: data.seed)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print summary accuracy only",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    run_masked_infer(
        cfg,
        samples=max(1, args.samples),
        split=args.split,
        seed=args.seed,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
