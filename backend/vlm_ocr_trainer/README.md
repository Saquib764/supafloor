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

Training randomly picks one canvas size per sample from `data.image_sizes` (width×height).

Sentence length is controlled by `data.sentence_len_min` / `data.sentence_len_max` (characters in the full transcription).

### Curriculum (easy → hard)

Start with all defaults, then widen one knob at a time. After each change, run a few thousand steps and inspect `training_session/viz/`.

| Setting | Default (easy) | When enabled | Impact |
| --- | --- | --- | --- |
| `line_counts` | `[1]` | `[1, 2, 3, …]` | More lines per image; model must segment and order multiple strings. |
| `variations.font` | `false` | `true` | Random printed font face per line instead of one fixed default. |
| `variations.font_size` | `false` | `true` | Random text height per line. |
| `variations.x_offset` | `false` | `true` | Random horizontal position per line (fixed left margin when off). |
| `variations.line_spacing` | `false` | `true` | Random vertical gap between lines. |
| `variations.background` | `false` | `true` | Off-white gray canvas instead of pure white. |
| `variations.ink` | `false` | `true` | Gray ink instead of black. |
| `variations.grain` | `false` | `true` | Pixel noise overlay (scan/photo-like). |
| `variations.handwritten` | `false` | `true` | Handwriting/script font pool — largest style shift; enable last. |

Suggested flag order: `font` → `font_size` → `x_offset` → `line_spacing` → `background` → `ink` → `grain` → `handwritten`.

With every flag `false`, samples render on a white canvas with black text at a fixed top-left origin, one default printed font, and uniform line spacing.

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
