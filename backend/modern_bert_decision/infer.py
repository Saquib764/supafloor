"""Forward and batched prediction helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from config import CONFIG_PATH, DecisionConfig, load_config
from device import pick_device
from formatting import format_batch
from model.decision_model import DecisionForwardOutput, ModernBERTDecisionModel


@dataclass(frozen=True)
class PredictionItem:
    selected_index: int
    logits: list[float]
    probabilities: list[float]
    options: list[str]


def forward_batch(
    model: ModernBERTDecisionModel,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    option_mask: torch.Tensor,
) -> DecisionForwardOutput:
    return model(input_ids, attention_mask, option_mask)


@torch.inference_mode()
def predict(
    model: ModernBERTDecisionModel,
    requests: list[dict[str, object]],
    *,
    device: torch.device,
) -> list[PredictionItem]:
    was_training = model.training
    model.eval()
    try:
        batch = format_batch(
            requests,
            cfg=model.cfg,
            tokenizer=model.tokenizer,
            max_len=model.max_len,
            device=device,
        )
        out = forward_batch(
            model,
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            option_mask=batch.option_mask,
        )
        results: list[PredictionItem] = []
        for row, req in enumerate(requests):
            options = [str(o) for o in req["options"]]  # type: ignore[index]
            mask = batch.option_mask[row].cpu()
            logits = out.logits[row].cpu()
            probs = out.probabilities[row].cpu()
            valid = mask.nonzero(as_tuple=False).squeeze(-1).tolist()
            if not valid:
                raise RuntimeError("internal error: row has no valid options")
            if isinstance(valid, int):
                valid = [valid]
            logits_list = [float(logits[i]) for i in valid]
            prob_list = [float(probs[i]) for i in valid]
            row_logits = logits.masked_fill(~mask, float("-inf"))
            sel = int(row_logits.argmax().item())
            results.append(
                PredictionItem(
                    selected_index=sel,
                    logits=logits_list,
                    probabilities=prob_list,
                    options=options,
                )
            )
        return results
    finally:
        if was_training:
            model.train()


def load_model(cfg: DecisionConfig, checkpoint: Path, device: torch.device) -> ModernBERTDecisionModel:
    meta = checkpoint / "decision_meta.json"
    if checkpoint.is_dir() and meta.is_file():
        from checkpoint import load_checkpoint

        model = load_checkpoint(checkpoint, cfg)
    else:
        from model.decision_model import build_model

        model = build_model(cfg)
    model.to(device)
    model.eval()
    return model


def parse_request_file(path: Path) -> list[dict[str, object]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "requests" in raw:
        return list(raw["requests"])
    if isinstance(raw, dict) and "options" in raw:
        return [raw]
    raise ValueError('request JSON must be {"state","question","options"} or {"requests":[...]}')


def request_from_race_example(cfg: DecisionConfig) -> list[dict[str, object]]:
    from data import example_to_request, load_race_preview_examples

    examples = load_race_preview_examples(cfg, count=1)
    if not examples:
        raise RuntimeError("no RACE rows available for --from-race")
    return [example_to_request(examples[0])]


def print_predictions(
    results: list[PredictionItem],
    requests: list[dict[str, object]],
    *,
    from_checkpoint: bool,
) -> None:
    for req, item in zip(requests, results, strict=True):
        question = str(req.get("question", ""))
        print(f"Question: {question}")
        for i, opt in enumerate(item.options):
            print(f"  [{i}] {opt}")
        print(
            json.dumps(
                {
                    "selected_index": item.selected_index,
                    "selected_option": item.options[item.selected_index],
                    "logits": item.logits,
                    "probabilities": item.probabilities,
                },
                indent=2,
            )
        )
        print(flush=True)
    if not from_checkpoint:
        print(
            "\nNote: no trained checkpoint loaded — scores are not meaningful decisions.",
            flush=True,
        )


def run_infer(
    cfg: DecisionConfig,
    *,
    checkpoint: Path,
    request_path: Path | None = None,
    from_race: bool = False,
    device: torch.device | None = None,
) -> list[PredictionItem]:
    device = device or pick_device()
    used_trained = (checkpoint / "decision_meta.json").is_file()
    model = load_model(cfg, checkpoint, device)
    if from_race:
        requests = request_from_race_example(cfg)
    elif request_path is not None:
        requests = parse_request_file(request_path)
    else:
        default = CONFIG_PATH.parent / "example_request.json"
        requests = parse_request_file(default)
    results = predict(model, requests, device=device)
    print_predictions(results, requests, from_checkpoint=used_trained)
    return results


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run decision-model inference")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint directory (default: infer.checkpoint in config.yaml)",
    )
    parser.add_argument(
        "--request",
        type=Path,
        default=None,
        help="JSON request file (default: example_request.json)",
    )
    parser.add_argument(
        "--from-race",
        action="store_true",
        help="Score one raw RACE validation example instead of a JSON file",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    checkpoint = args.checkpoint or cfg.infer.checkpoint
    run_infer(
        cfg,
        checkpoint=checkpoint,
        request_path=args.request,
        from_race=args.from_race,
    )


if __name__ == "__main__":
    main()
