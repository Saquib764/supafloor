# ModernBERT decision model

Multi-option scoring with a pretrained **ModernBERT** encoder and a CLS linear head. For each candidate option the model runs a separate forward pass on:

`[CLS] state … [SEP] question … [SEP] Is the correct answer to the question is: {option} …` (one forward per option)

It returns one logit per option (softmax over candidates). It does **not** generate text.

The CLS score head **must be trained** before predictions are useful.

Segments are delimited with the tokenizer’s existing `[SEP]` token (no extra vocab tokens). Context length is the encoder’s **8192** positions (`max_position_embeddings`); tokenization never truncates—overlong inputs raise during formatting. Scoring uses **CLS hidden state only**. Encoder adaptation: optional **LoRA** on encoder blocks (`train.lora.all_layers`, `train.lora.layers`, or auto), plus optional full-weight fine-tune on the last `train.encoder.last_n_layers` blocks. Score head is always a separate linear head (no LoRA). LoRA LR: `train.lora.lr`. Score head LR: `train.head.lr` (plain linear layer, no LoRA). Optional `[CLS]` row: `train.cls_embedding: {train, lr}`. Checkpoints always save `encoder/` (full base weights), `scoring_head.pt`, and tokenizer; when LoRA is trained, also `lora_adapter/`. Hub paths: `hub.cache`, `hub.pretrained`. Optimizer: AdamW via `train.optimizer` (`betas`, `eps`; weight decay from `train.schedule.weight_decay`). See `config.yaml` for the full nested layout (legacy flat keys under `train` still load).

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

- RACE rows are used as-is (no augmentation). Softmax uses every option in each row.
- Train: one row per question; encoder forward per option; **soft-target cross-entropy** on logits (one-hot at `answer_index`).
- **Test mode** — train on a fixed subset instead of the full split: set `train.test_mode.enabled: true` and `train.test_mode.samples` (e.g. 100 or 200), or run `uv run modern-bert-decision train --test-mode --test-samples 200`. The chosen rows are saved as `{session_dir}/test_train_samples.jsonl` and reused; pass `--test-refresh` to draw a new subset.
- Eval/infer: same multi-option forward; **softmax** over options; pick `argmax`.
- Checkpoints: `{session_dir}/last/` — full encoder, tokenizer, score head, optional LoRA adapter when trained, optional CLS row, and finetune toggles/LRs in meta (used on **resume**; **init_checkpoint** keeps weights but applies train settings from config).

## Tensor contract

| Tensor | Shape | Meaning |
|--------|--------|---------|
| `input_ids` | `[B, K_max, L]` | Padded token ids (one sequence per candidate) |
| `attention_mask` | `[B, K_max, L]` | 1 for real tokens |
| `option_mask` | `[B, K_max]` | True for real options |

Forward outputs: `logits`, `probabilities` `[B, K_max]`; `selected_indices` `[B]`.

## Checkpoints

Metadata includes `architecture: cls_per_candidate_v1`. Checkpoints from the old learnable segment-marker layout must be retrained.
