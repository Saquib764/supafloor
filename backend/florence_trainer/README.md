# Florence trainer

LoRA fine-tune of Microsoft Florence-2 so a floor-plan image maps to named room polygons.

Input is a realistic 1024×1024 PNG. Output is a list of `{ name, center, polygon }` decoded from Florence `<loc_*>` tokens.

This package is separate from `backend/floorplan/` so PyTorch / PEFT stay out of the geometry engine.

## Uses

- **Train** a LoRA adapter on a Hugging Face image→Florence dataset.
- **Infer** room polygons from a new floor-plan image using that adapter.
- **Not** for prompt→layout JSONL (`dev/export_training.py`). That is a different task.

## Dataset

Training loads Hub rows through `dataset_manager` (`data.dataset` in `config.yaml`, default `florence-rooms`). That view keeps only `stem`, `plan_id`, `variant`, `image`, and `florence` from `saquiboye/floorplan-to-json`. Train/val split is by **plan id** so variants of the same plan do not leak into validation.

Auth with `huggingface-cli login`, `HF_TOKEN`, or a `.env` in this folder. Do not put a token in `config.yaml`. Hub datasets are cached under `hf_cache/` in this folder.

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

Each room in the `.txt` target:

```
Living
<loc_158><loc_570>
<poly>
<loc_157><loc_569>
<loc_463><loc_569>
</poly>
<sep>
```

Name, center loc pair, then polygon vertices. Loc bins are 0–999. Decode uses `(loc + 0.5) / loc_bins * image_size`.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.11. GPU (CUDA) is recommended; `train.bf16` is applied only when CUDA is available.

```bash
cd backend/florence_trainer
uv sync
```

Edit `config.yaml` for model, Hub dataset, LoRA, and training knobs. Paths in the yaml are relative to this folder.

| Key | Meaning |
|---|---|
| `model_id` | HuggingFace Florence-2 checkpoint (`microsoft/Florence-2-base-ft` by default) |
| `task_prompt` | Prefix fed with the image (`<FLOORPLAN>`) |
| `data.dataset` | `dataset_manager` registry name (default `florence-rooms`) |
| `data.revision` | Optional Hub revision override |
| `lora.*` | Rank, alpha, dropout; vision encoder is frozen |
| `train.output_dir` | Adapter checkpoints (`checkpoints/floorplan-rooms/`, gitignored) |

## Execution

All commands from `backend/florence_trainer/`.

```bash
# Fine-tune LoRA
uv run florence-train train

# Optional: another config
uv run florence-train train --config config.yaml

# Image → rooms JSON on stdout
uv run florence-train infer --image /path/to/plan.png

# Write JSON; override adapter dir
uv run florence-train infer \
  --image /path/to/plan.png \
  --adapter checkpoints/floorplan-rooms \
  --out rooms.json
```

Infer loads the adapter from `train.output_dir` unless `--adapter` is set. The JSON is `{ "rooms": [...], "text": "<raw generation>" }`.

## Code

| File | Role |
|---|---|
| `cli.py` | `train` / `infer` entry (`uv run florence-train …`) |
| `config.yaml` / `config.py` | Training parameters |
| `loc.py` | Decode infer output (`parse_rooms`); labels come pre-formatted from the dataset |
| `data.py` | Hub dataset wrapper, plan-id split, collator |
| `model.py` | Load Florence-2, LoRA, custom tokens |
| `train.py` | HuggingFace `Trainer` loop (loads data via `dataset_manager.registry`) |
| `infer.py` | Generate text, parse rooms onto the input image size |
