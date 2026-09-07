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

Each training step picks `room_slots` rooms (with replacement if only one exists), builds `per_room_variation` prompt/label pairs per room (default 2 × 2 = 4), and with 5% probability per slot samples a point outside all rooms with an empty `<poly></poly>` label.

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
| `data.room_slots` | Rooms sampled per floorplan row (default 2) |
| `data.per_room_variation` | Referring-point samples per room (default 2) |
| `data.seed` | RNG seed for room/point selection per row |
| `data.dataset` | `dataset_manager` registry name (default `florence-rooms`) |
| `data.revision` | Optional Hub revision override |
| `lora.*` | Rank, alpha, dropout; vision encoder is frozen |
| `train.session_dir` | LoRA checkpoints (`training_session/`, gitignored) |
| `train.save_steps` | Save LoRA checkpoint every N steps (default 1000) |
| `train.num_workers` | DataLoader workers for parallel point sampling (default 4; set 0 to disable) |

## Execution

All commands from `backend/florence_trainer/`.

### Train

```bash
# Fine-tune LoRA
uv run florence-train train

# Preview model, LoRA, and random Hub samples
uv run florence-train inspect

# Optional: another config
uv run florence-train train --config config.yaml
```

Checkpoints are saved under `train.session_dir` (default `training_session/`) every `train.save_steps` steps, e.g. `training_session/checkpoint-12000/`.

### Infer

Inference loads a LoRA adapter directory (must contain `adapter_model.safetensors`). Default is `train.session_dir`; pass a checkpoint subfolder to use a specific step:

```bash
# e.g. checkpoint saved at step 12000
ADAPTER=training_session/checkpoint-12000
```

**Progressive grid inference** — samples bright grid points, infers room polygons iteratively, writes overlay to `dev/output/progressive-overlay.png`. Edit `_IMAGE`, `AUTOMATIC`, and the `adapter=` argument in the `run(...)` call at the bottom of `infer_progressive.py`, then:

```bash
uv run python infer_progressive.py
```

Set `AUTOMATIC = True` in that file to run the full grid without Enter prompts between attempts.

**Click-to-infer UI** — lists `test/` images on the left (sorted by name, first loaded by default); single mode infers on each click; multiple mode places points (and optional exclusions) then runs one inference when Infer is clicked. **UNet** and **VAE** run the visible image through the latest checkpoint from each trainer and replace the viewer image (repeat to chain); Florence always uses the current viewer image:

```bash
uv sync --group dev
uv run python dev/infer_ui.py --adapter "$ADAPTER"
```

Optional prep for `test/` images: `dev/fit_test_canvas.py` (fit to 1024×1024), `dev/pre_clean_test.py` (pre-clean copies).

Production automatic inference is wired through `backend/floorplan/extraction/florence.py` (adapter path in `extraction/florence_config.yaml`).

## Code

| File | Role |
|---|---|
| `florence_cli.py` | `train` / `inspect` entry (`uv run florence-train …`) |
| `config.yaml` / `config.py` | Training parameters |
| `loc.py` | Encode/decode loc tokens; parse infer/plot text |
| `data.py` | Hub loading, training/infer sample builders, collator |
| `model.py` | Load Florence-2, LoRA, custom tokens |
| `train.py` | HuggingFace `Trainer` loop (loads data via `dataset_manager.registry`) |
| `infer_progressive.py` | Progressive grid inference CLI |
| `dev/infer_ui.py` | Pygame UI — test image list + click-to-infer room polygons |
| `inference/` | Shared generate, grid sampling, and progressive inference |
