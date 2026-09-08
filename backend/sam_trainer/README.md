# SAM trainer

SAM 2.1 Hiera Small point-prompt segmentation for floor-plan images. Separate from `backend/floorplan/` so PyTorch stays out of the geometry engine.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.11. GPU (CUDA) or Apple Silicon (MPS) is recommended.

```bash
cd backend/sam_trainer
uv sync
```

Auth with `huggingface-cli login`, `HF_TOKEN`, or a `.env` in this folder. Weights download from `facebook/sam2.1-hiera-small` into `../hf_cache` on first run.

## Training

Training loads `floorplan-to-json` from `dataset_manager` (`saquiboye/floorplan-to-json`). Each row uses the realistic `image` at 1024×1024. For every floor plan the dataset picks random rooms, samples a random point inside each room polygon, and rasterizes that room as the mask target.

Rows are split into train/test via `data.test_fraction`. Every `train.viz_steps` steps the model runs on the held-out test rows and saves predicted mask overlays under `training_session/viz/step-XXXXXX/`.

Training always runs the live Hiera backbone with LoRA adapters on stages 3 and/or 4, plus any modules enabled in `finetune` (neck, prompt_encoder, mask_decoder).

```bash
uv run sam-train train
uv run sam-train inspect
```

| Key | Meaning |
|---|---|
| `data.dataset` | `dataset_manager` registry name (default `floorplan-to-json`) |
| `data.test_fraction` | Held-out test row fraction for viz (default `0.2`) |
| `data.room_slots` | Rooms sampled per floor plan row |
| `data.per_room_variation` | Point/mask pairs per room |
| `data.query_counts` | Possible query-point counts; one value is chosen per variation (e.g. `[1, 2, 3, 4]`) |
| `data.point_distance_alpha` | Inside-room point sampling weight ∝ distance-to-boundary^α (default `1.5`; `0` = uniform) |
| `finetune.neck` | Train FPN neck (~370K params) |
| `finetune.prompt_encoder` | Train point/mask prompt encoder |
| `finetune.mask_decoder` | Train mask decoder (~4.2M params) |
| `lora.rank` | LoRA rank for backbone adapters (alpha = 2× rank) |
| `lora.dropout` | LoRA dropout |
| `lora.stage3` | LoRA on Hiera stage 3 attention (`qkv`, `proj`) |
| `lora.stage4` | LoRA on Hiera stage 4 attention (`qkv`, `proj`) |
| `train.session_dir` | Checkpoints (`training_session/`, gitignored) |
| `train.resume` | Continue the same session (optimizer + step state) |
| `train.from_checkpoint` | Start a new session from a prior LoRA checkpoint; use a different `session_dir` |
| `train.viz_steps` | Save test-set mask overlays every N steps (default `10`) |
| `train.viz_samples` | Test rows to visualize each time (default `3`) |

Interactive click-to-segment inference lives in `backend/florence_trainer/dev/infer_ui.py` (toggle **Backbone: sam**).

## Code

| File | Role |
|---|---|
| `sam_cli.py` | `train` / `inspect` entry (`uv run sam-train …`) |
| `config.yaml` / `config.py` | Model, data, finetune, and train settings |
| `data.py` | Hub rows, room point sampling, mask rasterization |
| `canvas.py` | Fit images to canvas; mask overlay rendering (used by florence infer UI) |
| `model.py` | Load SAM 2.1, finetune flags, training forward, infer helpers |
| `train.py` | HuggingFace `Trainer` loop |
| `preview.py` | Print model structure + parameter counts by module |
