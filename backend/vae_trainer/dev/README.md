# Dev scripts

Manual utilities for inspecting floor-plan images through [TAESD](https://github.com/madebyollin/taesd) — the tiny autoencoder distilled for fast latent preview/decode.

Run from the package root (`backend/vae_trainer/`):

```bash
cd backend/vae_trainer

# Encode + decode every image in test_floorplans/ with FLUX.2 weights (taef2_*)
uv run python dev/flux2_roundtrip.py
```

On first run the script downloads `taef2_encoder.pth` and `taef2_decoder.pth` from the TAESD repo into `dev/weights/` (gitignored).

| Script | Purpose |
|---|---|
| `flux2_roundtrip.py` | FLUX.2 TAESD roundtrip on `test_floorplans/`; writes `dev/output/flux2-roundtrip/<stem>_roundtrip.jpg` |
| `taesd.py` | Vendored TAESD model code ([madebyollin/taesd](https://github.com/madebyollin/taesd), MIT) |

Each output JPEG is a horizontal triptych: **original | reconstruction | absolute error** (grayscale heatmap). The script prints per-image MSE/PSNR and latent spatial size (`input / 8` for FLUX.2's standard 8× architecture).

Edit `RESOLUTION` at the top of `flux2_roundtrip.py` to resize test images before encode/decode (default `256`, matching `config.yaml` `image_size`).
