"""Evaluate the saved checkpoint on the held-out prompt split."""

from __future__ import annotations

import os
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from config import CONFIG_PATH, Config, load_config
from data import PassageCollator, PassageExample, load_train_test_split
from device import get_device
from labels import (
    MODEL_FAMILIES,
    MODEL_NAMES,
    NUM_CLASSES,
    balanced_accuracy,
    family_for_label_index,
    index_to_model,
)
from model import ProbeClassifier, build_model
from train import CHECKPOINT_NAME


@dataclass(frozen=True)
class ScoredPrediction:
    row_id: str
    label_index: int
    model: str
    confidence: float
    top3_indices: tuple[int, int, int]
    cluster_label_index: int


def _kmeans_cosine(embeddings: torch.Tensor, k: int, *, iters: int = 50) -> torch.Tensor:
    """Unit-norm rows; returns cluster id in [0, k) per row."""
    n = int(embeddings.shape[0])
    if n < k:
        raise ValueError(f"need at least {k} embeddings for {k} clusters, got {n}")
    gen = torch.Generator(device=embeddings.device)
    gen.manual_seed(0)
    perm = torch.randperm(n, generator=gen, device=embeddings.device)
    centroids = F.normalize(embeddings[perm[:k]].clone(), dim=-1)
    assign = torch.zeros(n, dtype=torch.long, device=embeddings.device)
    for _ in range(iters):
        centroids = F.normalize(centroids, dim=-1)
        assign = (embeddings @ centroids.T).argmax(dim=-1)
        next_centroids: list[torch.Tensor] = []
        for cluster in range(k):
            mask = assign == cluster
            if not bool(mask.any().item()):
                pick = int(torch.randint(0, n, (1,), generator=gen, device=embeddings.device).item())
                next_centroids.append(embeddings[pick])
            else:
                mean = embeddings[mask].mean(dim=0)
                next_centroids.append(F.normalize(mean, dim=0))
        centroids = torch.stack(next_centroids, dim=0)
    return assign


def _cluster_labels_from_ce(
    assign: torch.Tensor, ce_labels: torch.Tensor, k: int
) -> torch.Tensor:
    """Per cluster, majority CE label; every row gets its cluster's label."""
    out = torch.empty(int(assign.shape[0]), dtype=torch.long, device=assign.device)
    for cluster in range(k):
        mask = assign == cluster
        if not bool(mask.any().item()):
            continue
        members = ce_labels[mask]
        label = int(torch.mode(members, dim=0).values.item())
        out[mask] = label
    return out


def _load_checkpoint(cfg: Config, path: Path) -> ProbeClassifier:
    if not path.is_file():
        raise FileNotFoundError(f"missing checkpoint {path}")
    from train import _load_model_weights

    model = build_model(cfg)
    if not _load_model_weights(model, path, cfg):
        raise RuntimeError(f"could not load checkpoint {path}")
    return model


@torch.no_grad()
def _predict(
    model: ProbeClassifier,
    examples: list[PassageExample],
    device: torch.device,
    batch_size: int,
) -> list[ScoredPrediction]:
    model.eval()
    collator = PassageCollator(
        model.tokenizer, max_len=model.max_len, probe_slots=model.num_probes
    )
    ce_labels: list[int] = []
    confidences: list[float] = []
    top3_list: list[tuple[int, int, int]] = []
    embed_chunks: list[torch.Tensor] = []
    steps = range(0, len(examples), batch_size)
    for start in tqdm(steps, desc="run-test", unit="batch"):
        chunk = examples[start : start + batch_size]
        batch = collator(chunk)
        ids = batch.input_ids.to(device)
        mask = batch.attention_mask.to(device)
        logits, contrast = model.forward_logits_and_contrastive(ids, mask)
        probs = F.softmax(logits.float(), dim=-1).cpu()
        embed_chunks.append(contrast.cpu())
        for row in probs:
            label = int(row.argmax())
            ce_labels.append(label)
            confidences.append(float(row[label]))
            top3_list.append(tuple(int(i) for i in row.topk(min(3, row.numel())).indices))
    embeddings = torch.cat(embed_chunks, dim=0)
    ce = torch.tensor(ce_labels, dtype=torch.long)
    assign = _kmeans_cosine(embeddings, NUM_CLASSES)
    cluster_labels = _cluster_labels_from_ce(assign, ce, NUM_CLASSES)
    out: list[ScoredPrediction] = []
    for ex, label, conf, top3, cluster_label in zip(
        examples, ce_labels, confidences, top3_list, cluster_labels.tolist(), strict=True
    ):
        out.append(
            ScoredPrediction(
                row_id=ex.row_id,
                label_index=label,
                model=index_to_model(label),
                confidence=conf,
                top3_indices=top3,
                cluster_label_index=int(cluster_label),
            )
        )
    return out


def _print_family_accuracy(gold: list[int], pred: list[int]) -> None:
    n = len(gold)
    family_correct = sum(
        1
        for g, p in zip(gold, pred, strict=True)
        if family_for_label_index(g) == family_for_label_index(p)
    )
    print(f"family accuracy: {family_correct / n:.4f} ({family_correct}/{n})", flush=True)
    print("family breakdown (gold rows → pred same family):", flush=True)
    for fam in MODEL_FAMILIES:
        gold_n = sum(1 for g in gold if family_for_label_index(g) == fam)
        if gold_n == 0:
            print(f"  {fam:8s}  n=   0", flush=True)
            continue
        ok = sum(
            1
            for g, p in zip(gold, pred, strict=True)
            if family_for_label_index(g) == fam and family_for_label_index(p) == fam
        )
        print(f"  {fam:8s}  family_acc={ok / gold_n:.4f}  ({ok}/{gold_n} gold rows)", flush=True)
    print(flush=True)


def _print_class_distribution_table(
    metric_rows: list[tuple[str, list[int]]],
    *,
    column_names: tuple[str, ...],
) -> None:
    n = len(column_names)
    if any(len(vals) != n for _, vals in metric_rows):
        raise ValueError("metric row length must match column_names")
    col_widths = [
        max(len(column_names[i]), *(len(str(vals[i])) for _, vals in metric_rows))
        for i in range(n)
    ]
    row_header = "model"
    label_w = max(len(row_header), *(len(name) for name, _ in metric_rows))
    gap = "  "
    header = "".join(
        column_names[i].rjust(col_widths[i]) + gap for i in range(n)
    )
    print(f"{row_header:<{label_w}}{gap}{header.rstrip()}", flush=True)
    rule = gap.join("-" * col_widths[i] for i in range(n))
    print(f"{'-' * label_w}{gap}{rule}", flush=True)
    for metric, vals in metric_rows:
        cells = "".join(str(vals[i]).rjust(col_widths[i]) + gap for i in range(n))
        print(f"{metric:<{label_w}}{gap}{cells.rstrip()}", flush=True)


def _print_class_bias(gold: list[int], predictions: list[ScoredPrediction]) -> None:
    pred = [int(p.label_index) for p in predictions]
    gold_c = Counter(gold)
    pred_c = Counter(pred)
    correct_c = Counter(g for g, p in zip(gold, pred, strict=True) if g == p)
    top3_c = Counter(
        g for g, p in zip(gold, predictions, strict=True) if g in p.top3_indices
    )
    cluster = [int(p.cluster_label_index) for p in predictions]
    cluster_c = Counter(cluster)
    cluster_correct_c = Counter(
        g for g, c in zip(gold, cluster, strict=True) if g == c
    )
    metric_columns = ("gold", "pred", "correct", "top3", "cluster", "cluster_ok")
    per_class = [
        gold_c,
        pred_c,
        correct_c,
        top3_c,
        cluster_c,
        cluster_correct_c,
    ]
    model_rows = [
        (
            name,
            [counts.get(i, 0) for counts in per_class],
        )
        for i, name in enumerate(MODEL_NAMES)
    ]
    print("class distribution (one row per model):", flush=True)
    _print_class_distribution_table(model_rows, column_names=metric_columns)
    print(flush=True)


def _print_mean_confidence(label: str, confs: list[float]) -> None:
    if confs:
        mean = sum(confs) / len(confs)
        print(f"mean confidence ({label}): {mean:.4f}  (n={len(confs)})", flush=True)
    else:
        print(f"mean confidence ({label}): n/a  (n=0)", flush=True)


def _print_predicted_count_std(pred: list[int]) -> None:
    pred_c = Counter(pred)
    counts = [pred_c.get(i, 0) for i in range(NUM_CLASSES)]
    if not counts:
        return
    std = statistics.pstdev(counts) if len(counts) > 1 else 0.0
    mean = sum(counts) / len(counts)
    print(
        f"predicted count std dev (per class): {std:.4f}  "
        f"mean={mean:.2f}  (n_classes={len(counts)}, rows={len(pred)})",
        flush=True,
    )


def _print_summary(predictions: list[ScoredPrediction], examples: list[PassageExample]) -> None:
    print("\n=== Whole passage (Qwen3 probe) ===", flush=True)
    print(f"rows: {len(predictions)}", flush=True)
    if not predictions:
        return
    gold = [int(ex.label_index) for ex in examples]
    pred = [int(p.label_index) for p in predictions]
    correct = sum(1 for g, p in zip(gold, pred, strict=True) if g == p)
    labeled = len(examples)
    print(f"accuracy:       {correct / labeled:.4f} ({correct}/{labeled})", flush=True)
    top3_hit = sum(
        1
        for ex, p in zip(examples, predictions, strict=True)
        if int(ex.label_index) in p.top3_indices
    )
    print(f"top-3 accuracy: {top3_hit / labeled:.4f} ({top3_hit}/{labeled})", flush=True)
    bal = balanced_accuracy(gold, pred)
    print(
        f"balanced accuracy: {bal:.4f}  (mean per-model recall, {NUM_CLASSES} classes)",
        flush=True,
    )
    cluster = [int(p.cluster_label_index) for p in predictions]
    cluster_correct = sum(1 for g, c in zip(gold, cluster, strict=True) if g == c)
    print(
        f"clustering accuracy: {cluster_correct / labeled:.4f} "
        f"({cluster_correct}/{labeled})  "
        f"(18 cosine k-means on contrastive emb, label = dominant CE per cluster)",
        flush=True,
    )
    cluster_bal = balanced_accuracy(gold, cluster)
    print(
        f"clustering balanced accuracy: {cluster_bal:.4f}  "
        f"(mean per-model recall, {NUM_CLASSES} classes)",
        flush=True,
    )
    _print_family_accuracy(gold, pred)
    _print_class_bias(gold, predictions)
    correct_confs: list[float] = []
    incorrect_confs: list[float] = []
    top3_hit_confs: list[float] = []
    top3_miss_confs: list[float] = []
    for ex, p in zip(examples, predictions, strict=True):
        gold_i = int(ex.label_index)
        if gold_i == int(p.label_index):
            correct_confs.append(p.confidence)
        else:
            incorrect_confs.append(p.confidence)
        if gold_i in p.top3_indices:
            top3_hit_confs.append(p.confidence)
        else:
            top3_miss_confs.append(p.confidence)
    _print_mean_confidence("correct", correct_confs)
    _print_mean_confidence("incorrect", incorrect_confs)
    _print_mean_confidence("top3 hit", top3_hit_confs)
    _print_mean_confidence("top3 miss", top3_miss_confs)
    _print_predicted_count_std(pred)
    print(flush=True)


def evaluate_model(
    model: ProbeClassifier,
    examples: list[PassageExample],
    device: torch.device,
    batch_size: int,
    *,
    step: int | None = None,
) -> None:
    if not examples:
        print("run-test: test split is empty; skipped", flush=True)
        return
    n_prompts = len({ex.prompt_id for ex in examples})
    step_bit = f"step {step} | " if step is not None else ""
    print(
        f"\nrun-test: {step_bit}rows={len(examples)} prompts={n_prompts}",
        flush=True,
    )
    was_training = model.training
    preds = _predict(model, examples, device, batch_size)
    _print_summary(preds, examples)
    model.train(was_training)


def run_test(
    cfg: Config,
    *,
    device: torch.device | None = None,
    checkpoint: Path | None = None,
) -> None:
    _train_ex, test_ex = load_train_test_split(cfg)
    if not test_ex:
        raise ValueError("test split is empty; set data.test_fraction > 0 to evaluate a holdout")
    path = (checkpoint or (cfg.train.session_dir / CHECKPOINT_NAME)).resolve()
    print(
        f"run-test: {cfg.data.test_csv} prompt holdout | {cfg.model_id} | "
        f"test_fraction={cfg.data.test_fraction} seed={cfg.data.seed}",
        flush=True,
    )
    print(f"checkpoint: {path}", flush=True)
    device = device or get_device(announce=True)
    model = _load_checkpoint(cfg, path)
    model.to(device)
    evaluate_model(model, test_ex, device, cfg.train.batch_size)


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Qwen3 authorship model on held-out prompts")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    run_test(cfg, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
