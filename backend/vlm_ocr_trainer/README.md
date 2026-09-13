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

# Generate word collections then render holdout paragraphs
uv run vlm-ocr-train words
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

Training picks one canvas size per **batch** from the current step's `data.image_sizes` (width×height); all samples in that batch share the same H×W. Shorter text sequences are padded with `<PAD>` only — images are not spatially padded.

Training samples use the same paragraph pipeline as holdout: words, numbers, and punctuation from `renderer/words.json` are composed into 1–3 paragraphs and rendered with font/size/opacity/background/grain variation. Labels contain only text that fit on the canvas.

Tune `data.paragraph_counts` and `paragraph_len_max` in `config.yaml` (defaults match `holdout.*`). Font size is random 8–16 px per sample.

## Holdout benchmark

Training data is infinite synthetic text, but accuracy is measured on a **fixed holdout** written to disk:

1. Run `uv run vlm-ocr-train words` — writes `renderer/words.json` (topic → word lists + punctuation).
2. Run `uv run vlm-ocr-train holdout` — samples words, numbers, and punctuation into 1–3 paragraphs, renders with font/size/opacity/background/grain/padding variation. Manifest labels contain **only text that fit on the canvas**.
3. During training, every `train.benchmark_steps` optimizer steps the model runs greedy decode on the holdout.
4. Metrics logged: **CER**, **WER**, **seq acc** (exact string match).

Tune `words.*` and `holdout.paragraph_counts`, `paragraph_len_max` in `config.yaml`.

`train.save_best: true` writes `training_session/best.pt` when holdout CER improves. Per-epoch metrics append to `training_session/metrics.jsonl`.

## Output layout

```
renderer/
└── words.json

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
