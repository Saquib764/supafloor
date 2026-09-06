# UNet trainer

Frozen [TAESD](https://github.com/madebyollin/taesd) (taef2) around a trainable compressor + ViT + expander. Encodes a 1-channel floor-plan `image` to a taef2 latent, compresses it, runs a ViT with a residual skip around it, upsamples, and decodes to `simple_image`.

Set `image_size` in `config.yaml` (512–1024, divisible by `16 × vit.patch_size`). Default 512. taef2 is 8× / 32-ch; the compressor halves spatial size and maps 32→16 channels.

Each Hub row yields two pairs: `image → simple_image` and `simple_image → simple_image`. Both columns are converted to grayscale and resized to `image_size`. The target is always `simple_image`.

This package is separate from `backend/floorplan/` so PyTorch stays out of the geometry engine.

## Loss

Ink-weighted L1 reconstruction (taef2 encoder/decoder stay frozen; no KL):

```
w = 1 + α · (1 − x)
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

Checkpoints write `training_session/last.pt` (trainable weights only). Every `recon_steps`, a 2×3 JPEG is saved as `training_session/recon/step-NNNNNN.jpg`: top `image | recon | target`, bottom `simple_image | recon | target`.

## Code

| File | Role |
|---|---|
| `unet_cli.py` | `train` / `inspect` entry (`uv run unet-train …`) |
| `config.yaml` / `config.py` | Training parameters |
| `taesd.py` | Vendored TAESD (taef2 encoder/decoder stay frozen) |
| `model.py` | Compressor, ViT, expander, reconstruction loss |
| `data.py` | Hub `image` / `simple_image` → 1×image_size×image_size tensors |
| `train.py` | PyTorch loop, checkpoints, recon dumps |
| `preview.py` | Print model + dummy forward |
