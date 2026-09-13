# OcrVlm — ~1M-parameter character OCR

Grayscale text image in, character transcription out. A miniature TrOCR-style model: CNN stem → ViT encoder → autoregressive text decoder. No separate text detector or CTC head.

See `uv run vlm-ocr-train inspect` for live architecture, parameter counts, and a sample I/O export to `_preview/`.

All architecture constants, validation, grid math, and inspect output live in `model.py`.

**Target size:** ~0.9M parameters (run `inspect` for current count).

---

## End-to-end flow

```
Synthetic grayscale image (random W×H canvas, printed/handwritten text)
    │
    ▼  image_to_tensor()
normalize: x = pixel / 127.5 − 1   →  float32 [1, H, W] in [-1, 1]
    │
    ▼  OcrVlm.encode_image()   (computed once at inference)
visual tokens [1, (W/4)·(H/8), 112]     256×256 → 64×32 = 2048 tokens
    │
    ▼  OcrVlm.generate()  or  teacher-forced forward()
character ids  [<BOS>, R, o, o, m, …, <EOS>]
    │
    ▼  CharVocab.decode()
transcription string  "Room 203\nWidth: 4.5m"
```

Training uses on-the-fly synthetic paragraph pairs — no external dataset.

---

## Input

| Property | Value |
|---|---|
| Channels | 1 (grayscale) |
| Canvas sizes | From `data.image_sizes` schedule (e.g. 128×128, 256×256, 256×128, 128×256); one size per batch |
| Normalization | `(pixel / 127.5) - 1` → range `[-1, 1]` |
| Tensor shape | `[B, 1, H, W]` — all images in a batch share the same H×W |
| Text padding | Shorter token sequences padded with `<PAD>`; ignored in loss |

**Constraint:** W divisible by 4, H divisible by 8 (`STEM_STRIDE_W`, `STEM_STRIDE_H` in `model.py`).

### Stem compression

Anisotropic CNN stem (W÷4, H÷8). Example grids:

| Canvas (W×H) | Visual grid | Tokens |
|---|---|---|
| 256×256 | 64×32 | 2048 |
| 256×128 | 64×16 | 1024 |
| 128×256 | 32×32 | 1024 |
| 128×128 | 32×16 | 512 |

---

## Architecture

```
Input: [B, 1, H, W]
        │
        ▼
CnnStem (W÷4, H÷8)
        │
        ▼
[B, 96, H/8, W/4]
        │
        ▼ flatten
[B, (H/8·W/4), 96]
        │
        ▼
+ fixed 2D sinusoidal pos
        │
        ▼
VisionEncoder × 8   (d=96, 4 heads, MLP=192, H/V axial attn, 2D RoPE)
        │
        ▼
LayerNorm + Linear 96→112
        │
        ▼
visual tokens [B, N_vis, 112]     N_vis = (H/8)·(W/4)
        │
        │ cross-attention (fixed during decode)
        ▼
TextDecoder × 2     (d=112, 4 heads, MLP=256, causal self-attn + cross-attn, 1D RoPE)
        │
        ▼
lm_head (weight-tied to token_embedding)
        │
        ▼
[B, seq, 113] logits
```

### CNN stem

Conv stride 1, conv stride 2, dwsep stride 2 (÷4), three local blocks, vertical-only downsample, two local blocks, 1×1 projection.

| Stage | Op | Channels | Spatial W×H (256×256 in) |
|---|---|---|---|
| 0 | Conv 3×3, stride 1 | 1 → 16 | 256×256 |
| 1 | Conv 3×3, stride 2 | 16 → 24 | 128×128 |
| 2 | DepthwiseSep, stride 2 | 24 → 48 | 64×64 |
| 3–5 | Residual local block ×3 | 48 → 48 | 64×64 |
| 6 | DepthwiseSep, stride (2,1) | 48 → 72 | **64×32** |
| 7–8 | Residual local block ×2 | 72 → 72 | 64×32 |
| 9 | Conv 1×1 | 72 → 96 | **64×32** |

---

## Vocabulary

Character-level — 113 tokens (ASCII + research symbols like α, β, μ, ±, ≤, ≥, ×, °, ², –, —, curly quotes; BOS/EOS/PAD). Defined in `vocab.py`.

---

## Training & inference

Teacher-forced cross-entropy; greedy autoregressive decode at inference. See `train.py` and `model.generate()`.

---

## Module reference

```
vlm_ocr_trainer/
├── model.py      OcrVlm, CnnStem, STEM_STRIDE_W/H, visual_grid, inspect helpers
├── vocab.py      CharVocab
├── data.py       SyntheticOcrDataset, OcrCollator, HoldoutDataset
├── config.yaml   image sizes, training hyperparams
└── cli.py        vlm-ocr-train train|inspect
```
