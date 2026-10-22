# Gemma authorship classifier (probe)

`google/gemma-3-270m` with max length 512. Each row is tokenized as the passage text only. Tokenization leaves room for `num_probes` learnable probe tokens (default 10) (each initialized as the mean embedding of *what*, *why*, and *how*), appended in fixed order after the passage. At each tapped layer, all 10 probe hidden states are concatenated; tap vectors are concatenated and fed to the linear head. Supervised contrastive loss uses a separate two-layer `contrast_head` (configurable hidden/output dims) on the pooled vector, then L2-normalizes that projection. With `train.freeze_blocks: true` (default), Gemma blocks stay frozen and only probe + head train. Set `freeze_blocks: false` to fine-tune all 18 blocks at `train.block_lr`. Token embeddings stay frozen unless `train.freeze_embeddings: false` (only when blocks train). Tapped block outputs at the probe position are LayerNormed, concatenated, and classified into the 18 Name That LLM labels.

`train.freeze_probes` / `train.freeze_head` fix probe embeddings or the pool-norm + linear head. Probes use `train.probe_lr`; the head uses `train.head_lr`. At least one of `freeze_blocks`, `freeze_probes`, or `freeze_head` must be false. Each optimizer step samples `train.prompt_count` prompt ids and the same `batch_size / prompt_count` models from each. `train.micro_batch_size` splits the forward so it fits on an Apple MPS GPU. Set `train.micro_batch: false` to run the whole batch in one forward.

## Setup

```bash
cd backend/gemma_author_probe
uv sync
```

Gemma weights are gated. Put `HF_TOKEN` in `.env`, or reuse `backend/decision_gemma/.env`. `train.csv` is read from `data/`.

## Commands

```bash
uv run gemma-author-probe preview --config-only   # head and labels, no weight download
uv run gemma-author-probe preview                  # also load the hub checkpoint
uv run gemma-author-probe train --max-steps 1      # one optimizer step on MPS
uv run gemma-author-probe run-test                 # held-out prompts from training_session_probe/last.pt
```

A full run is `uv run gemma-author-probe train`. Checkpoints are `session_dir/last.pt` and `session_dir/training_state.pt`. Set `train.checkpoint.resume: true` to continue the same `session_dir`.

Block-finetune checkpoints from `backend/gemma_author` are not compatible (no `probe` key).
