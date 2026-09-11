# OcrVlm — ~1M-parameter character OCR

Grayscale text image in, character transcription out. A miniature TrOCR-style model: CNN stem → ViT encoder → autoregressive text decoder. No separate text detector or CTC head.

See `uv run vlm-ocr-train inspect` for live architecture, parameter counts, and a sample I/O export to `_preview/`.

**Target size:** ~0.93M parameters (929,800 trainable).

---

## End-to-end flow

```
Synthetic grayscale image (random W×H canvas, printed/handwritten text)
    │
    ▼  image_to_tensor()
normalize: x = pixel / 127.5 − 1   →  float32 [1, H, W] in [-1, 1]
    │
    ▼  OcrVlm.encode_image()   (computed once at inference)
visual tokens [1, (H/16)·(W/16), 128]
    │
    ▼  OcrVlm.generate()  or  teacher-forced forward()
character ids  [<BOS>, R, o, o, m, …, <EOS>]
    │
    ▼  CharVocab.decode()
transcription string  "Room 203\nWidth: 4.5m"
```

Training uses on-the-fly synthetic pairs from `data.render_text_sample()` — no external dataset.

---

## Input

| Property | Value |
|---|---|
| Channels | 1 (grayscale) |
| Canvas sizes | 256×256, 256×128, 128×256, 128×128 (width×height, random per sample) |
| Normalization | `(pixel / 127.5) - 1` → range `[-1, 1]` |
| Tensor shape | `[B, 1, H, W]` |
| Batch padding | Pad to max H×W in batch with `1.0` (white) |

**Constraint:** H and W must be divisible by 16 (four stride-2 downsamples in the CNN stem).

### Synthetic label

Random OCR-style lines (room labels, dimensions, IDs, words). Reading order: top→bottom, left→right; lines separated by `\n`.

```
Room 203
Width: 4.5m
```

Encoded as:

```
<BOS> R o o m <SPACE> 2 0 3 <NEWLINE> W i d t h : <SPACE> 4 . 5 m <EOS>
```

Sentence length clamped by `data.sentence_len_min` / `data.sentence_len_max` (characters in full transcription).

---

## Architecture

```
Input: [B, 1, H, W]
        │
        ▼
CnnStem (4× downsample, SiLU + GroupNorm)
        │
        ▼
[B, 96, H/16, W/16]
        │
        ▼ flatten
[B, (H/16·W/16), 96]
        │
        ▼
VisionEncoder × 4   (d=96, 4 heads, MLP=192, bidirectional, 2D RoPE)
        │
        ▼
LayerNorm + Linear 96→128
        │
        ▼
visual tokens [B, N_vis, 128]     N_vis = (H/16)·(W/16)
        │
        │ cross-attention (fixed during decode)
        ▼
TextDecoder × 3     (d=128, 4 heads, MLP=256, causal self-attn + cross-attn, 1D RoPE)
        │
        ▼
lm_head (weight-tied to token_embedding)
        │
        ▼
[B, seq, 99] logits
```

### CNN stem

| Stage | Op | Channels | Spatial (256×256 in) |
|---|---|---|---|
| 0 | Conv 3×3, stride 2 | 1 → 24 | 128×128 |
| 1 | DepthwiseSep, stride 2 | 24 → 48 | 64×64 |
| 2 | DepthwiseSep, stride 2 | 48 → 72 | 32×32 |
| 3 | DepthwiseSep, stride 2 | 72 → 96 | 16×16 |

Depthwise separable block: DWConv 3×3 → PWConv 1×1 → GroupNorm(1) → SiLU.

### Vision encoder

- **4 layers**, pre-LayerNorm Transformer blocks
- **d_model = 96**, **4 heads** (head dim 24), **MLP = 192**
- **Bidirectional** self-attention (all image tokens see each other)
- **2D RoPE:** first half of head dim rotated by y-position, second half by x-position

### Vision → language projection

Single linear layer: `96 → 128` (no bias). Output visual tokens stay fixed for the entire decode loop.

### Text decoder

- **3 layers**, pre-LayerNorm
- **d_model = 128**, **4 heads** (head dim 32), **MLP = 256**
- Each layer: causal self-attention → cross-attention to visual tokens → MLP
- **1D RoPE** on text self-attention
- **Weight tying:** `token_embedding.weight == lm_head.weight`

---

## Vocabulary

Character-level — no BPE or SentencePiece.

| Group | Count |
|---|---|
| a–z, A–Z | 52 |
| 0–9 | 10 |
| ASCII punctuation | 32 |
| space, newline | 2 |
| `<BOS>`, `<EOS>` | 2 |
| `<PAD>` (batching only) | 1 |
| **Total** | **99** |

Defined in `vocab.py` (`CharVocab`).

---

## Training

**Objective:** teacher-forced next-token cross-entropy.

At timestep t, predict `P(y_t | image, y_0 … y_{t-1})`. Positions tagged `<PAD>` are ignored.

| Setting | Default |
|---|---|
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 0.01 |
| Warmup | 3% of total steps, then cosine decay |
| Grad clip | 1.0 |
| Max sequence length | 128 |

Default training data is synthetic (`SyntheticOcrDataset`): `samples_per_epoch` random pairs per epoch, no disk cache.

---

## Inference

Greedy autoregressive decode via `OcrVlm.generate()`:

1. `visual = encode_image(image)` — run once, cache for all steps
2. `generated = [<BOS>]`
3. Loop: `logits = decode(visual, generated)` → `next = argmax(logits[:, -1])` → append
4. Stop at `<EOS>` or `max_seq_len`
5. `vocab.decode(generated)` → string

No beam search initially. KV caching for text self-attention and cross-attention K/V is not yet implemented (full re-decode each step).

---

## Parameter budget

| Component | Params |
|---|---|
| CNN stem | ~13.5K |
| Vision encoder (4 layers) | ~297.8K |
| Vision projection | ~12.3K |
| Text decoder (3 layers) | ~593.5K |
| Token embedding (tied lm_head) | ~12.7K |
| **Total** | **~929.8K** |

Layer-wise breakdown: `uv run vlm-ocr-train inspect`.

---

## Spatial shapes (examples)

| Canvas (W×H) | Stem output | Visual tokens N_vis |
|---|---|---|
| 256×256 | 96 × 16 × 16 | 256 |
| 256×128 | 96 × 16 × 8 | 128 |
| 128×256 | 96 × 8 × 16 | 128 |
| 128×128 | 96 × 8 × 8 | 64 |

Mixed-size batches are padded to the largest H×W before the forward pass.

---

## Module reference

```
vlm_ocr_trainer/
├── model.py      OcrVlm, CnnStem, VisionEncoderBlock, DecoderBlock, RoPE1D/2D, ocr_loss
├── vocab.py      CharVocab, charset, encode/decode
├── data.py       render_text_sample, SyntheticOcrDataset, OcrCollator, pad_images_to_batch
├── train.py      training loop, viz export
├── preview.py    inspect + _preview/ sample export
├── canvas.py     input | output viz panels
├── config.yaml   image sizes, sentence length, training hyperparams
└── cli.py        vlm-ocr-train train|inspect
```

---

## Output layout

```
training_session/
├── last.pt
└── viz/
    └── step-000200/
        ├── sample-000.jpg    # input | predicted text
        └── sample-001.jpg

_preview/                     # inspect --sample export
├── input.jpg
├── gt.jpg
├── output.jpg
└── preview.jpg
```
