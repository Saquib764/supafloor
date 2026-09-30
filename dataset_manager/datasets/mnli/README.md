# MNLI (decision format)

[Multi-Genre Natural Language Inference](https://cims.nyu.edu/~sbowman/multinli/) (Williams et al.), exported for the ModernBERT decision trainer.

Each row is one three-way choice:

| Field | Content |
|-------|---------|
| `state` | `Premise:` / `Hypothesis:` labels, each followed by the text on the next line |
| `question` | Fixed: relationship between premise and hypothesis |
| `options` | Entailment / neutral / contradiction phrasing (fixed order) |
| `answer_index` | `0` entailment, `1` neutral, `2` contradiction |

Hub source: [`nyu-mll/glue`](https://huggingface.co/datasets/nyu-mll/glue) config `mnli`.

## Download JSONL splits

```bash
cd dataset_manager
uv run python cli.py download mnli
```

Writes under repo-root `mnli-data/`:

```
mnli-data/train.jsonl
mnli-data/validation_matched.jsonl
mnli-data/validation_mismatched.jsonl
```

## Training config

Set `data.dataset: mnli`, `data.train_split: train`, and e.g. `data.val_split: validation_matched` in `backend/modern_bert_decision/config.yaml`.
