# ModernBERT decision model

Multi-option scoring with a pretrained **ModernBERT** encoder and a CLS linear head. For each candidate option the model runs a separate forward pass on:

`[CLS] [state] … [question] … [options] … [candidate] …` (one forward per option; candidate text is the option under test)

It returns one logit per option (softmax over candidates). It does **not** generate text.

The CLS score head **must be trained** before predictions are useful.

Four learnable segment markers — `[state]`, `[question]`, `[options]`, `[candidate]` — are added to the tokenizer; configure each under `train.markers.<name>: {train, lr}`. Scoring uses **CLS hidden state only**. Partial encoder fine-tune: `train.encoder.last_n_layers` and `train.encoder.layers[]`. Optional `[CLS]` row: `train.cls_embedding: {train, lr}`. Hub paths: `hub.cache`, `hub.pretrained`. See `config.yaml` for the full nested layout (legacy flat keys under `train` still load).

## Setup

```bash
cd backend/modern_bert_decision
uv sync
```

Hub cache defaults to `../hf_cache` (see `config.yaml`).

## Commands

```bash
uv run modern-bert-decision inspect
uv run modern-bert-decision verify
uv run modern-bert-decision download-data
uv run modern-bert-decision train
uv run modern-bert-decision infer
```

## Data and training

- RACE rows are used as-is (no option subsetting or shuffling augment).
- Train: dataset length = examples × `variations`; slot 0 = gold candidate, slots 1… = wrong (shuffled); BCE 0/1.
- Eval/infer: one forward per option; **softmax** over options; pick `argmax`.
- Checkpoints: `{session_dir}/last/` — full encoder, tokenizer, score head, optional CLS row, and finetune toggles/LRs in meta (used on **resume**; **init_checkpoint** keeps weights but applies train settings from config).

## Tensor contract

| Tensor | Shape | Meaning |
|--------|--------|---------|
| `input_ids` | `[B, K_max, L]` | Padded token ids (one sequence per candidate) |
| `attention_mask` | `[B, K_max, L]` | 1 for real tokens |
| `option_mask` | `[B, K_max]` | True for real options |

Forward outputs: `logits`, `probabilities` `[B, K_max]`; `selected_indices` `[B]`.

## Checkpoints

Metadata includes `architecture: cls_per_candidate_v1`. Checkpoints from the old marker-token layout must be retrained.
