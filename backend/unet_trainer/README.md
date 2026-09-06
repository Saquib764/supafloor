# UNet trainer

Trainable compressor + ViT + expander maps floor plans (`image` → `simple_image`). By default (`train.precompute_latent: true`), taef2 encode runs in `data.py`, the model trains in latent space, and taef2 decode is used only for recon JPEG dumps. Set `precompute_latent: false` to run taef2 inside the model and train with image-space loss (previous behavior).

Set `image_size` in `config.yaml` (256–1024, divisible by `16 × vit.patch_size`). Default 512. taef2 is 8× / 32-ch; the compressor halves spatial size and maps 32→16 channels.

Each Hub row yields two pairs: `image → simple_image` and `simple_image → simple_image`. With precompute enabled, both columns are encoded once per row and the `simple_image` latent is reused for the identity pair.

This package is separate from `backend/floorplan/` so PyTorch stays out of the geometry engine.

## Loss

Ink-weighted L1 (encoder/decoder frozen; no KL). Space depends on `train.precompute_latent`:

**Precomputed latents (`true`, default):**

```
w = 1 + α · (1 − x)   # x = simple_image, downsampled to latent resolution
L = mean(w · |ẑ − z|)
```

**In-model taef2 (`false`):**

```
w = 1 + α · (1 − x)   # x = simple_image pixels
L = mean(w · |x̂ − x|)
```

`α` is `loss.ink_weight` (default 10).

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.11. GPU (CUDA) is recommended; `train.bf16` is applied only when CUDA is available.

```bash
cd backend/unet_trainer
uv sync
```

Auth with `huggingface-cli login`, `HF_TOKEN`, or a `.env` in this folder. Training streams `image` and `simple_image` from the Hub (`data.dataset: floorplan-vae`) so Arrow does not copy the full dataset to disk. It does not read local export folders.

Hub cache is `../hf_cache` (shared with `vae_trainer`). First `inspect`/`train` downloads `taef2_encoder.pth` and `taef2_decoder.pth` into `../hf_cache/taesd/` if they are missing.

Edit `config.yaml` for dataset, ViT, loss, and training knobs. Paths in the yaml are relative to this folder. At 1024, set `vit.patch_size: 2` unless you have VRAM for 4096 tokens.

## Execution

All commands from `backend/unet_trainer/`.

```bash
# Print architecture, param count, and dummy-forward shapes
uv run unet-train inspect

# Train
uv run unet-train train
```

Checkpoints write `training_session/last.pt` (trainable weights only). Every `recon_steps`, a 2×3 JPEG is saved as `training_session/recon/step-NNNNNN.jpg`: top `image | recon | target`, bottom `simple_image | recon | target`. Recon panels are decoded from predicted latents via frozen taef2.

## Code

| File | Role |
|---|---|
| `unet_cli.py` | `train` / `inspect` entry (`uv run unet-train …`) |
| `config.yaml` / `config.py` | Training parameters |
| `taesd.py` | Vendored TAESD (taef2 weights; data precompute or in-model when `precompute_latent: false`) |
| `model.py` | Compressor, ViT, expander; optional taef2 wrap; latent and image losses |
| `data.py` | Hub images → tensors; optional taef2 latent precompute |
| `train.py` | PyTorch loop, checkpoints, recon dumps |
| `preview.py` | Print model + dummy forward |
