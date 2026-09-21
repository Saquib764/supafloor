# Inference core

FPN floor plan structure mask inference, optional Replicate image-edit cleaning, and manual labeling tools.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.11. Uses weights from `backend/fpn_five_trainer/`.

```bash
cd backend/inference_core
uv sync
```

For pygame UIs (`infer_ui_simple.py`, `label_data.py`), install dev dependencies:

```bash
uv sync --group dev
```

Set `OPENAI_API_KEY` in `.env` when running step-001 room-name detection or step-003 LLM polygon refinement (not required for `infer-ui-simple` when the key is absent — step-001 skips the room-name LLM).

## Image edit + room mask discovery

Fits the input to a 1024×1024 white canvas, runs three chained Replicate image-edit steps (structure clean → close openings → remove furniture), saves each step, then finds room masks as connected white regions in the grayscaled final image.

```bash
uv run infer-image-edit --image test_image/fp01.jpg
```

Use `--skip-replicate` to reuse saved `03-no-furniture.png`.

Outputs land in `output/image_edit/<stem>/` (`00-canvas.png` … `03-no-furniture.png`, plus `*-diff.png` per step) with room masks in `masks/gray.png`, `masks/step-NNN.png`, and `masks/all-masks.png`.

Set `REPLICATE_API_TOKEN` or `replicate_token` in `config.yaml`.

## `infer-fpn` — FPN Five mask inference

CLI entry point for the three-step FPN pipeline. Fits the input to a square canvas, runs FPN Five (background / rooms / walls / doors / windows), optionally re-infers on a cropped region, and optionally refines polygons with an LLM.

```bash
uv run infer-fpn --image test_image/BE_01.jpg
# or
uv run python infer_fpn.py --image test_image/BE_01.jpg
```

### Pipeline

| Step | What it does |
|---|---|
| **step-001** | Detect floorplan region from background mask, mask the full canvas, run FPN inference, LLM room-name detection, clean masks, fit polygon JSON, wall centerlines |
| **step-002** | Crop to region bbox (+ padding), re-infer on the cropped canvas, same artifacts as step-001 |
| **step-003** | Send input + room mask to the LLM; write refined room/door/window polygons and `polygon-overlay.jpg` |

Default run executes all three steps. Use flags to stop early or skip the crop pass.

### CLI flags

| Flag | Default | Description |
|---|---|---|
| `--config` | `config.yaml` | Config path |
| `--image` | `test_image/fp01.jpg` | Floor plan image |
| `--checkpoint` | `fpn.checkpoint_latest` | Override FPN weights path |
| `--output-dir` | `fpn.output_dir` | Base output directory; writes `<output-dir>/<stem>/step-NNN/` |
| `--skip-step-2` | off | Skip crop re-infer; step-003 uses the step-001 canvas |
| `--skip-step-3` | off | Skip LLM refinement; final outputs come from step-002 (or step-001 if `--skip-step-2`) |

### Common invocations

Full pipeline (default):

```bash
uv run infer-fpn --image real_floorplan/BE_01_001.jpg
```

Mask inference only — no LLM (used by `infer-ui-simple`):

```bash
uv run infer-fpn --image test_image/BE_01.jpg --skip-step-3
```

LLM on full canvas — skip crop re-infer (used by `label_data` in `llm` mode):

```bash
uv run infer-fpn --image real_floorplan/BE_01_001.jpg --skip-step-2
```

Step-001 only:

```bash
uv run infer-fpn --image test_floorplan/BE_01.jpg --skip-step-2 --skip-step-3
```

Custom checkpoint and output location:

```bash
uv run infer-fpn \
  --image test_image/BE_01.jpg \
  --checkpoint ../fpn_five_trainer/training_session_2/last.pt \
  --output-dir output/fpn/old
```

Canvas size defaults to `fpn.size` (768 px). Override in code via `run_fpn_pipeline(..., canvas_size=1024)`.

Step-003 requires `OPENAI_API_KEY` in `.env` and `fpn.room_polygon_llm.*` in config.

### Output layout

Outputs land in `<output-dir>/<stem>/`:

| Step | Key artifacts |
|---|---|
| **step-001** | `floorplan-region-mask.jpg`, `floorplan-region.json`, `input-masked.jpg`, `masks.jpg`, `masks-clean.jpg`, `room-names.json`, `room-mask-overlays.jpg`, `mask-polygon-comparison.jpg`, `structure-polygons/`, `final-masks/` |
| **step-002** | `input-cropped.jpg`, `input-masked.jpg`, same artifact set as step-001 (skipped with `--skip-step-2`) |
| **step-003** | `input.jpg`, `room-polygons.jpg`, `floorplan-1024.json`, `final-masks/` with `polygon-overlay.jpg` (skipped with `--skip-step-3`) |

Each `final-masks/` directory contains PNG masks and fitted polygon JSON per layer:

- `background.png`, `room.png`, `wall.png`, `door.png`, `window.png`
- `room.json`, `wall.json`, `door.json`, `window.json`

Step-001 and step-002 polygons are fit directly from FPN masks. Step-003 polygons are LLM-refined and include a translucent preview overlay.

## Simple FPN comparison UI

Side-by-side viewer for comparing **old** vs **latest** checkpoints on images in `test_image/`. Runs step-001 and step-002 only (`--skip-step-3`); no OpenAI key required.

```bash
uv run infer-ui-simple
# or
uv run python infer_ui_simple.py
```

Each run writes under `output/fpn/old/<stem>/` or `output/fpn/latest/<stem>/`. The viewer shows three rows per checkpoint:

| Row | Source |
|---|---|
| Wall-subtract | Middle panel of `step-002/room-mask-overlays.jpg` |
| Skimage skeleton | `step-002/structure-polygons/wall-centerline-skimage-skeleton.jpg` |
| Mask polygons | Fitted polygons from `step-002/final-masks/*.json` on `step-002/input-masked.jpg` |

Use **Run old** / **Run latest** to infer the selected image; **Clear** removes both variant output trees for that stem.

## Manual labeling

Interactive pygame UI to refine structure masks. Source images and output paths are set under `label_data` in `config.yaml` (defaults: `real_floorplan/`, `floorplan-manual-mask/`).

```bash
uv run label-data
# or
uv run python label_data.py
```

### What gets saved

| Location | Contents |
|---|---|
| `<label_data.output_dir>/input/<id>.jpg` | Source fit to 1024×1024 (unmasked) |
| `<label_data.output_dir>/output/<id>/` | Mask PNGs (white = active) + `polygon-overlay.jpg` |
| `<label_data.pipeline_output_dir>/<id>/step-001/final-masks/` or `step-003/final-masks/` | Polygon JSON edited in place + preview images (see `label_data.mode`) |

Wall is mask-only (brush or **Compute wall**). Room, door, window, and background use polygon edit. Masks are derived from polygons on save.

Label-data inference runs the FPN pipeline at **1024 px**. Default **`label_data.mode: normal`** runs step-001 only (fitted mask polygons, no LLM). Set **`mode: llm`** for step-001 + step-003 LLM refinement (step-002 skipped).

### Batch auto-labeling

Run processed inference for every image in `label_data.input_dir` (no UI):

```bash
uv run python label_data.py --auto
```

Skips images that already have complete outputs (all mask PNGs under `<label_data.output_dir>/output/<id>/` plus `room.json` under the mode’s final-masks dir). Re-run everything with:

```bash
uv run python label_data.py --auto --force
```

## Config

| Key | Meaning |
|---|---|
| `canvas_size` | Default fit canvas for image-edit path |
| `replicate_token` | Replicate API token (overridden by `REPLICATE_API_TOKEN`) |
| `image_edit.model` | Replicate image-edit model id |
| `image_edit.output_dir` | Base directory for per-image cleaning outputs |
| `image_edit.prompts.*` | Prompts for the cleaning steps |
| `room_masks.white_threshold` | Grayscale value treated as white |
| `room_masks.min_area_px` | Minimum connected white region area to keep |
| `fpn.size` | Default FPN inference canvas side length (px) |
| `fpn.checkpoint_latest` | Path to fpn_five_trainer weights |
| `fpn.output_dir` | Base directory for FPN pipeline outputs |
| `fpn.mask_clean.*` | Morphological open/close and min area for mask cleanup |
| `fpn.floorplan_region.*` | Background-based region detection settings |
| `fpn.room_components.min_area_px` | Minimum room component area |
| `fpn.room_polygon_llm.*` | Step-003 LLM model, size, and cost settings |
| `label_data.mode` | `normal` (step-001 only, default) or `llm` (step-001 + step-003) for label-data inference |
| `label_data.input_dir` | Source floor plan images to label (default `real_floorplan/`, relative to `inference_core/`) |
| `label_data.output_dir` | Manual label dataset root (default `floorplan-manual-mask/`, relative to repo root); writes `input/` and `output/` beneath it |
| `label_data.pipeline_output_dir` | FPN pipeline cache for label-data inference (default `output/label_data/`) |
