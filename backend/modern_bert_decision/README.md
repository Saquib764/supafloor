# ModernBERT decision model

Multi-option scoring with a pretrained **ModernBERT** encoder and one shared linear head. The model encodes `[CLS] [STATE] … [QUESTION] … [OPTION] option_i … [SEP]` in a single forward pass and returns one logit (and softmax probability) per option. It does **not** generate text.

The scoring head and embeddings for `[STATE]`, `[QUESTION]`, and `[OPTION]` are new and **must be trained** before predictions are useful. Untrained runs only demonstrate wiring and tensor shapes.

Toggle fine-tuning of the original encoder with `model.train_pretrained_encoder` in `config.yaml`:

| Value | Behavior |
|-------|----------|
| `false` (default) | Pretrained encoder frozen; train marker embeddings + score head |
| `true` | Full encoder trainable (all layers and vocab rows) |

Legacy key `model.freeze_encoder` is still read as the inverse of `train_pretrained_encoder`.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.11. Works on CPU, CUDA, or Apple MPS (no FlashAttention requirement).

```bash
cd backend/modern_bert_decision
uv sync
```

Hub cache defaults to `../hf_cache` (see `config.yaml`). Downloads use that path explicitly (`cache_dir`), not only `~/.cache/huggingface`. The encoder needs ~650 MiB free on that volume.

If a download fails mid-way, delete the partial tree  
`hf_cache/hub/models--answerdotai--ModernBERT-base` and retry after freeing space.

## Commands

```bash
# Architecture, parameter counts, dummy forward shapes
uv run modern-bert-decision inspect

# Hand-written checks (shapes, padding, gradients, save/reload, reject oversize)
uv run modern-bert-decision verify

# Download RACE to repo-root race-data/ (train.jsonl, validation.jsonl, test.jsonl)
uv run modern-bert-decision download-data

# Train (streams from local JSONL when data.source is local or auto + files exist)
uv run modern-bert-decision train

# Infer (uses config infer.checkpoint + example_request.json by default)
uv run modern-bert-decision infer
uv run modern-bert-decision infer --from-race
uv run modern-bert-decision infer --request example_request.json
uv run modern-bert-decision infer --checkpoint training_session/last

# Same as infer:
uv run python infer.py
```

Example request file:

```json
{
  "state": "The kitchen is east of the living room.",
  "question": "Which option is the pantry door?",
  "options": ["Door A", "Door B"]
}
```

## Data and training

- **`data`**: `dataset: race`, Hub `source: hub` (or local CMU extract under repo `race-data/`). See `dataset_manager/datasets/race/`.
- **`train`**: session dir, LR, batch size, `save_steps`, `resume`, etc.
- Loss: cross-entropy over option logits vs `answer_index`.
- Checkpoints: `{session_dir}/last/` — by default **light** saves (`train.save_full_encoder: false`): tokenizer, scoring head, and marker embedding rows only (~few MB). Reload merges with Hub encoder weights. Set `save_full_encoder: true` to also write the full encoder (~600 MB; needs free disk).
- `training_state.pt` stores optimizer state for resume.

| Key | Role |
|-----|------|
| `session_dir` | Checkpoints written to `{session_dir}/last/` |
| `init_checkpoint` | Optional prior checkpoint directory (weights only) |
| `resume` | Reload optimizer state from `session_dir/last` when training exists |
| `save_steps` | Checkpoint interval (steps) |

## Tensor contract

| Tensor | Shape | Meaning |
|--------|--------|---------|
| `input_ids` | `[B, L]` | Padded token ids |
| `attention_mask` | `[B, L]` | 1 for real tokens |
| `option_positions` | `[B, K_max]` | Sequence index of each `[OPTION]` marker; 0 where padded |
| `option_mask` | `[B, K_max]` | True for real options |

Forward outputs (batched):

| Field | Shape |
|-------|--------|
| `logits` | `[B, K_max]` (padded slots `-inf`) |
| `probabilities` | `[B, K_max]` (softmax in float32; padded slots 0) |
| `selected_indices` | `[B]` (index into each request’s `options` list) |

Build batches with `formatting.format_batch` or use `infer.predict` for a list of request dicts.

## Checkpoints

Save (directory layout):

```python
from model.decision_model import build_model
from config import load_config

model = build_model(load_config())
model.save_pretrained("training_session/last")
```

Reload:

```python
from model.decision_model import ModernBERTDecisionModel
from config import load_config

model = ModernBERTDecisionModel.load_pretrained("training_session/last", load_config())
```

Contents:

- `encoder/` — full encoder weights (including resized token embeddings)
- `tokenizer/` — tokenizer with marker tokens
- `scoring_head.pt` — trained `Linear(H, 1)` weights
- `decision_meta.json` — marker ids, context limit, architecture metadata

Loading a checkpoint **requires** a successful restore of encoder and scoring head; it does not fall back to random initialization.

## Code layout

| File | Role |
|------|------|
| `decision_cli.py` | `inspect` / `verify` / `predict` |
| `config.yaml` / `config.py` | Hub id, marker strings, paths |
| `formatting.py` | Token layout, padding, validation |
| `model/decision_model.py` | Encoder + shared head |
| `checkpoint.py` | Save/load |
| `infer.py` | `forward_batch`, `predict` |
| `preview.py` | Model summary |
| `verify.py` | Architecture checks |
