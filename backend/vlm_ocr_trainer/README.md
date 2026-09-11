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

# Train on synthetic text images
uv run vlm-ocr-train train
```

## Data config

Training randomly picks one canvas size per sample from `data.image_sizes` (width×height):

- 256×256, 256×128, 128×256, 128×128

Sentence length is controlled by `data.sentence_len_min` / `data.sentence_len_max` (characters in the full transcription).

## Output layout

```
training_session/
├── last.pt
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
