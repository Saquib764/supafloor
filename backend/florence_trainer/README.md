# Florence trainer

LoRA fine-tune of Microsoft Florence-2 so a floor-plan image maps to the room polygon containing a referring point.

Input is a realistic 1024×1024 PNG plus a loc-token point. Output is one room polygon and its name (or `unlabeled`).

This package is separate from `backend/floorplan/` so PyTorch / PEFT stay out of the geometry engine.

## Uses

- **Train** a LoRA adapter on a Hugging Face image→Florence dataset.
- **Infer** room polygons from a new floor-plan image using that adapter.
- **Not** for prompt→layout JSONL (`dev/export_training.py`). That is a different task.

## Dataset

Training loads Hub rows through `dataset_manager` (`data.dataset` in `config.yaml`, default `florence-rooms`). That view keeps `stem`, `plan_id`, `variant`, `image`, and `extraction` from `saquiboye/floorplan-to-json`. Loc-token labels are built from `extraction` at train time (not stored on the Hub). Room polygons in extraction JSON are canonicalized on export: clockwise winding, first vertex closest to the origin on the 1024 canvas.

Each training step picks two rooms (or the same room twice when only one exists), builds four prompt/label pairs per floorplan (two point variations per room slot), and with 5% probability per slot samples a point outside all rooms with an empty `<poly></poly>` label.

Auth with `huggingface-cli login`, `HF_TOKEN`, or a `.env` in this folder. Do not put a token in `config.yaml`. Hub cache is `../hf_cache` (shared with `vae_trainer` and `unet_trainer`).

To publish the full Hub dataset, export locally from `backend/floorplan/` then upload via `dataset_manager/` (registry name `floorplan-to-json`):

```bash
uv run python dev/export_image_to_floorplan.py \
  --dir ../../floorplan_dataset/basic_5000 \
  --out ../../image-to-floorplan/basic_5000

cd ../../dataset_manager
uv sync
uv run python cli.py upload floorplan-to-json
```

### Label format

Each training sample:

**Prompt (input):**
```
Locate room polygon containing point <loc_602><loc_955>
```

**Label (target):**
```
<poly><loc_120><loc_180><loc_420><loc_180><loc_420><loc_460><loc_120><loc_460></poly>
Room name: Bath
```

When the render omitted the room text label, the name line is `Room name: unlabeled`.

Loc bins are 0–999 on a 1024 canvas. Decode uses `(loc + 0.5) / loc_bins * image_size`.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.11. GPU (CUDA) is recommended; `train.bf16` is applied only when CUDA is available.

```bash
cd backend/florence_trainer
uv sync
```

Edit `config.yaml` for model, Hub dataset, LoRA, and training knobs. Paths in the yaml are relative to this folder.

| Key | Meaning |
|---|---|
| `model_id` | HuggingFace Florence-2 checkpoint (`microsoft/Florence-2-base` by default) |
| `canvas_size` | Coordinate space for loc tokens (1024, matches extraction JSON) |
| `data.seed` | RNG seed for room/point selection per row |
| `data.dataset` | `dataset_manager` registry name (default `florence-rooms`) |
| `data.revision` | Optional Hub revision override |
| `lora.*` | Rank, alpha, dropout; vision encoder is frozen |
| `train.session_dir` | LoRA checkpoints (`training_session/`, gitignored) |
| `train.save_steps` | Save LoRA checkpoint every N steps (default 1000) |

## Execution

All commands from `backend/florence_trainer/`.

```bash
# Fine-tune LoRA
uv run florence-train train

# Optional: another config
uv run florence-train train --config config.yaml

# Image → all room polygons (iterative referring segmentation)
uv run florence-train infer --image /path/to/plan.png

# Write JSON; override adapter dir
uv run florence-train infer \
  --image /path/to/plan.png \
  --adapter training_session \
  --out rooms.json
```

Infer loads the adapter from `train.session_dir` unless `--adapter` is set. Inference samples `infer.num_points` random canvas points (default 16), runs one batched forward pass, and merges unique room polygons into plot-ready output:

```
Bath
<poly><loc_120><loc_180>...</poly>

Kitchen
<poly><loc_520><loc_520>...</poly>
```

The JSON is `{ "rooms": [...], "text": "<plot format>", "prompts": [...] }`.

## Code

| File | Role |
|---|---|
| `florence_cli.py` | `train` / `infer` entry (`uv run florence-train …`) |
| `config.yaml` / `config.py` | Training parameters |
| `loc.py` | Encode/decode loc tokens; parse infer/plot text |
| `data.py` | Hub loading, training/infer sample builders, collator |
| `model.py` | Load Florence-2, LoRA, custom tokens |
| `train.py` | HuggingFace `Trainer` loop (loads data via `dataset_manager.registry`) |
| `infer_progressive.py` | Progressive grid inference CLI |
