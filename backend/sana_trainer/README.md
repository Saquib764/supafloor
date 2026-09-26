# sana_trainer

Fine-tune [Sana 0.6B diffusers](https://huggingface.co/Efficient-Large-Model/Sana_600M_1024px_diffusers) on generic **image + caption** datasets.

Training follows the NVIDIA diffusers recipe ([`train_dreambooth_lora_sana.py`](https://github.com/NVlabs/Sana/blob/main/train_scripts/train_dreambooth_lora_sana.py)):

- **Default:** LoRA on the transformer (`lora.rank`, `lora.alpha` with **alpha = rank**).
- **Toggle:** `train.full_finetune: true` or `sana-train train --full-finetune` trains **transformer weights only** (VAE + Gemma text encoder frozen).
- **Single GPU** via Hugging Face Accelerate (`gradient_accumulation_steps` from config).

## Setup

```bash
cd backend/sana_trainer
cp .env.example .env   # optional HF_TOKEN
uv sync
```

If `SanaTransformer2DModel` is missing, install a newer diffusers build:

```bash
uv pip install "git+https://github.com/huggingface/diffusers.git"
```

## Config

Edit `config.yaml`:

| Section | Notes |
|--------|--------|
| `data.dataset` | Hub id (e.g. `diffusers/dog-example`) or local dataset directory |
| `data.image_column` / `data.caption_column` | Column names in the split |
| `lora.*` | LoRA rank, alpha, dropout, `target_modules` |
| `train.full_finetune` | `false` = LoRA, `true` = full transformer |
| `train.viz_steps` / `train.viz_prompts` | Periodic sample images under `session_dir/viz/` |

## Commands

```bash
uv run sana-train preview
uv run sana-train preview --skip-image
uv run sana-train train
uv run sana-train train --full-finetune
uv run sana-train train --from-adapter training_session/lora/final
```

Checkpoints:

- LoRA: `training_session/lora/step-XXXXXX/` and `training_session/lora/final/`
- Full: `training_session/checkpoints/step-XXXXXX/transformer/` and `training_session/transformer/`
- LoRA resume (optimizer state): `training_session/checkpoint-XXXX/` with `train.resume: true`
