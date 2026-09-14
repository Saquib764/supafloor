# vlm_ocr_trainer

~1M-parameter vision-language OCR model: **grayscale image → character tokens**.

Model architecture: [MODEL.md](MODEL.md)

Training data is generated on the fly — random text lines are rendered into synthetic images with matching transcriptions. No external dataset is required.

## Setup

```bash
cd backend/vlm_ocr_trainer
uv sync
```

## Commands

```bash
# Print architecture, parameter count, and export sample I/O to _preview/
uv run vlm-ocr-train inspect

# Skip _preview/ export
uv run vlm-ocr-train inspect --no-sample

# Generate holdout dataset
uv run vlm-ocr-train holdout
uv run vlm-ocr-train holdout --force

# Train on synthetic text images
uv run vlm-ocr-train train

# Benchmark a checkpoint on the holdout set
uv run vlm-ocr-train benchmark
uv run vlm-ocr-train benchmark --checkpoint training_session/best.pt
```

## Model

CNN stem uses anisotropic W÷4, H÷8 (`STEM_STRIDE_W`, `STEM_STRIDE_H` in `model.py`): 256×256 canvas → 64×32 visual grid (2048 tokens). Existing checkpoints are incompatible after architecture changes — retrain from scratch.

## Data config

Training picks one canvas size and one `max_seq_len` per **DataLoader batch** from the current step's schedules; all samples in that batch share the same H×W and length budget (including with `num_workers > 0`). Shorter text sequences are padded with `<PAD>` only — images are not spatially padded.

Training samples randomly draw characters from the vocabulary charset, render them with font/size/opacity/background/grain variation, and label only the text that fit on the canvas.

Tune `data.max_text_len` in `config.yaml` (defaults match `holdout.max_text_len`). Font size is random 8–16 px per sample.

## Holdout benchmark

Training data is infinite synthetic text, but accuracy is measured on a **fixed holdout** written to disk:

1. Run `uv run vlm-ocr-train holdout` — renders random charset text with font/size/opacity/background/grain/padding variation. Manifest labels contain **only text that fit on the canvas**.
2. During training, every `benchmark.steps` optimizer steps the model runs greedy decode on the holdout.
3. Metrics logged: **CER**, **WER**, **seq acc** (exact string match).

Tune `holdout.max_text_len` and `benchmark.*` in `config.yaml`. `benchmark.max_seq_len` caps greedy decode length on holdout (separate from training `max_seq_len`). Set `benchmark.num_samples` (e.g. `64`) for a quick eval; omit it to use the full holdout.

`benchmark.save_best: true` writes `training_session/best.pt` when holdout CER improves. Per-epoch metrics append to `training_session/metrics.jsonl`.

## Output layout

```
holdout/
├── manifest.jsonl
└── images/
    └── 000000.png

training_session/
├── last.pt
├── best.pt
├── metrics.jsonl
└── viz/
    └── step-000200/
        ├── sample-000.jpg   # input | output
        └── sample-001.jpg

_preview/                    # inspect export
├── input.jpg
├── gt.jpg
├── output.jpg
└── preview.jpg              # input | output
```

Checkpoints write `training_session/last.pt`. Every `viz_steps`, overlays land in `training_session/viz/step-NNNNNN/sample-NNN.jpg`: each file is `input | output`.
