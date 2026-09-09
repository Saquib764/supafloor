# FPN trainer

Star-prompted room segmentation for floor plans. A red filled star marks the target room on the input image; a custom CNN + YOLO-like FPN predicts that room's boundary mask.

Separate from `backend/floorplan/` so PyTorch stays out of the geometry engine.

Model architecture, module explanations, and size variants: **[MODEL.md](MODEL.md)**.

## Task

| | |
|---|---|
| Input | Floor-plan → grayscale → invert → RGB (white walls on black) + red star |
| Output | 3-channel mask: R=boundary, G=doors, B=windows (RGB for viz) |
| Dataset | `floorplan-to-json` (Hub stream) |
| Resolution | `input_size: 512` (Hub images resized; extraction coords scaled from 1024 canvas) |

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.11. GPU (CUDA) or Apple Silicon (MPS) recommended.

```bash
cd backend/fpn_trainer
uv sync
```

Auth with `huggingface-cli login`, `HF_TOKEN`, or a `.env` in this folder. Hub cache is `../hf_cache`.

## Execution

```bash
# Print architecture, param count, dummy-forward shapes
uv run fpn-train inspect

# Train
uv run fpn-train train

# Infer (star coords in original image pixels)
uv run fpn-train infer --image test_image/001.jpeg --cx 400 --cy 300 --radius 10
```

Checkpoints write `training_session/last.pt`. Every `viz_steps`, test-set overlays land in `training_session/viz/step-NNNNNN-<stem>.jpg`: `input | pred | target`.

## Loss

Dice + focal on the room mask logits (no box loss).

## Code

| File | Role |
|---|---|
| `fpn_cli.py` | `train` / `inspect` / `infer` entry |
| `config.yaml` / `config.py` | Training parameters |
| `star.py` | Filled 5-point star drawing |
| `data.py` | Hub stream, scaled extraction, star samples |
| `net/` | CNN backbone, YOLO-like FPN, mask head, loss |
| `train.py` | PyTorch loop, checkpoints, viz |
| `preview.py` | Model summary + dummy forward |
| `infer.py` | Single-image inference |
| `canvas.py` | Mask overlay rendering |
