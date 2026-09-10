# FPN five-class trainer

Floor-plan structure segmentation: grayscale schematic in, five mask channels out (background, walls, rooms, doors, windows). Custom CNN + YOLO-like FPN.

Separate from `backend/floorplan/` so PyTorch stays out of the geometry engine.

Model architecture, module explanations, and size variants: **[MODEL.md](MODEL.md)**.

## Task

| | |
|---|---|
| Input | Floor-plan → grayscale → invert (white walls on black), single channel |
| Output | 5-channel mask: background / walls / rooms / doors / windows |
| Dataset | `floorplan-to-json` (Hub stream) |
| Resolution | `input_size: [512, 768]` (Hub images resized; extraction coords scaled from 1024 canvas) |

Channel order:

| Index | Name | Source |
|---|---|---|
| 0 | background | `1 - clip(walls + rooms)` |
| 1 | walls | `fpn_trainer` wall rasterizer |
| 2 | rooms | `fpn_room_trainer` room interior rasterizer |
| 3 | doors | `fpn_trainer` opening rasterizer |
| 4 | windows | `fpn_trainer` opening rasterizer |

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.11. GPU (CUDA) or Apple Silicon (MPS) recommended.

```bash
cd backend/fpn_five_trainer
uv sync
```

Auth with `huggingface-cli login`, `HF_TOKEN`, or a `.env` in this folder. Hub cache is `../hf_cache`.

## Execution

```bash
# Print architecture, param count, dummy-forward shapes
uv run fpn-train inspect

# Train
uv run fpn-train train

# Infer
uv run fpn-train infer --image test_image/001.jpeg
```

Checkpoints write `training_session/last.pt`. Every `viz_steps`, test-set overlays land in `training_session/viz/step-NNNNNN/<stem>.jpg`: `input | pred | target`.

## Loss

Per-channel dice + focal with `channel_weights`. Wall channel gets an opening false-negative penalty; room channel gets an opening false-positive penalty (same semantics as `fpn_trainer` + `fpn_room_trainer`).

## Code

| File | Role |
|---|---|
| `fpn_cli.py` | `train` / `inspect` / `infer` entry |
| `config.yaml` / `config.py` | Training parameters |
| `data.py` | Hub stream, scaled extraction, five-class targets |
| `net/` | CNN backbone, YOLO-like FPN, mask head, loss |
| `train.py` | PyTorch loop, checkpoints, viz |
| `preview.py` | Model summary + dummy forward |
| `infer.py` | Single-image inference |
| `canvas.py` | Mask overlay rendering |
