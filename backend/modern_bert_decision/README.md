# ModernBERT decision model

Multi-option scoring with a pretrained **ModernBERT** encoder and a shared MLP score head (LayerNorm → Linear → GELU → Linear). Each question is **one sequence**: context, question, instruction, then each option as `{option} : [MASK]` with `[SEP]` boundaries and `[CLS]` after `[SEP]` on option segments (see `formatting.py`).

The model gathers the hidden state at each option’s configured token (`model.score_head.token`: `mask` or `cls`), runs **`model.decision_layers.count`** small transformer layers over those option hiddens (self-attention across options), then applies the MLP score head and softmaxes over options. It does **not** generate text.

The score head **must be trained** before predictions are useful.

Sequence length is capped by **`model.max_len`** (≤ encoder `max_position_embeddings`); longer inputs are **truncated** (context first). All option `[MASK]` tokens must remain in the sequence. Encoder adaptation: optional full-weight fine-tune on the last `train.encoder.last_n_layers` blocks. Decision stack and score head train from scratch. Checkpoints save `encoder/`, `decision_stack.pt`, `scoring_head.pt`, and tokenizer. Architecture metadata: `mask_token_decision_stack_v1`.

## Setup

```bash
cd backend/modern_bert_decision
uv sync
```

## Commands

```bash
uv run modern-bert-decision inspect
uv run modern-bert-decision verify
uv run modern-bert-decision download-data
uv run modern-bert-decision train
uv run modern-bert-decision infer
uv run modern-bert-decision masked-infer   # vanilla MLM baseline (untrained head)
```

## Data and training

- Registry datasets (`race`, `mnli`, …) → `state` / `question` / `options` / `answer_index`.
- Train: **one encoder forward** per question; **soft-target cross-entropy** on option logits (one-hot at `answer_index`).
- **Test mode** — `train.test_mode.enabled` or `--test-mode --test-samples N`.
- Checkpoints: `{session_dir}/last/`. Legacy `mean_pool_per_candidate_v1` checkpoints must be retrained.

## Tensor contract

| Tensor | Shape | Meaning |
|--------|--------|---------|
| `input_ids` | `[B, L]` | One masked sequence per question |
| `attention_mask` | `[B, L]` | 1 for real tokens |
| `score_index` | `[B, K_max]` | Index of each option’s score token (`mask` or `cls`) |
| `option_mask` | `[B, K_max]` | True for real options |

Forward outputs: `logits`, `probabilities` `[B, K_max]`; `selected_indices` `[B]`.
