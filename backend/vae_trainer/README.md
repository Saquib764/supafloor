# VAE trainer

Small convolutional VAE that encodes a 1-channel floor-plan `image` to a spatial latent and decodes it to `simple_image`.

Set `image_size` and `patch_size` in `config.yaml`. Latent spatial size is `image_size / patch_size`. Defaults are 256×256×1 with `patch_size: 32` → 8×8×4. Hub images are resized to `image_size`.

Each Hub row yields two pairs: `image → simple_image` and `simple_image → simple_image`. Both columns are converted to grayscale and resized to `image_size`.

This package is separate from `backend/floorplan/` so PyTorch stays out of the geometry engine.

## Loss

Ink-weighted L1 reconstruction plus a small KL to a unit Gaussian:

```
w = 1 + α · (1 − x)
L = mean(w · |x̂ − x|) + β · KL(q(z|x) ‖ N(0, I))
```

`α` is `loss.ink_weight` (default 10). `β` is `loss.kl_beta` (default `1e-5`), linearly warmed up over `kl_warmup_steps`.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.11. GPU (CUDA) is recommended; `train.bf16` is applied only when CUDA is available.

```bash
cd backend/vae_trainer
uv sync
```

Auth with `huggingface-cli login`, `HF_TOKEN`, or a `.env` in this folder. Training streams `image` and `simple_image` from the Hub (`data.dataset: floorplan-vae`) so Arrow does not copy the full dataset to disk. It does not read local export folders.

Edit `config.yaml` for dataset, loss, and training knobs. Paths in the yaml are relative to this folder. Hub cache is `../hf_cache` (shared with `unet_trainer`).

## Execution

All commands from `backend/vae_trainer/`.

```bash
# Print architecture, param count, and dummy-forward shapes
uv run vae-train inspect

# Train
uv run vae-train train
```

Checkpoints write `training_session/last.pt`. Every `recon_steps`, a 2×3 JPEG is saved as `training_session/recon/step-NNNNNN.jpg`: top `image | recon | target`, bottom `simple_image | recon | target`.

## Code

| File | Role |
|---|---|
| `cli.py` | `train` / `inspect` entry (`uv run vae-train …`) |
| `config.yaml` / `config.py` | Training parameters |
| `model.py` | Encoder, decoder, reparameterization, loss |
| `data.py` | Hub `image` / `simple_image` → 1×image_size×image_size tensors |
| `train.py` | PyTorch loop, checkpoints, recon dumps |
| `preview.py` | Print model + dummy forward |
