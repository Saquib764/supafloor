# Inference core

FPN floor plan structure mask inference, optional Replicate image-edit cleaning, and manual labeling tools.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.11. Uses weights from `backend/fpn_five_trainer/`.

```bash
cd backend/inference_core
uv sync
```

For the manual labeling UI (`label_data.py`), install dev dependencies (pygame):

```bash
uv sync --group dev
```

Set `OPENAI_API_KEY` in `.env` when running step-003 LLM polygon refinement.

## Image edit + room mask discovery

Fits the input to a 1024×1024 white canvas, runs three chained Replicate image-edit steps (structure clean → close openings → remove furniture), saves each step, then finds room masks as connected white regions in the grayscaled final image.

```bash
uv run infer-image-edit --image test_image/fp01.jpg
```

Use `--skip-replicate` to reuse saved `03-no-furniture.png`.

Outputs land in `output/image_edit/<stem>/` (`00-canvas.png` … `03-no-furniture.png`, plus `*-diff.png` per step) with room masks in `masks/gray.png`, `masks/step-NNN.png`, and `masks/all-masks.png`.

Set `REPLICATE_API_TOKEN` or `replicate_token` in `config.yaml`.

## FPN Five mask inference

Three-step pipeline: region detection + FPN inference on the full canvas (step-001), optional crop re-infer (step-002), then LLM room/door/window polygon refinement (step-003).

```bash
uv run infer-fpn --image real_floorplan/BE_01_001.jpg
```

Skip step-002 (run step-003 directly on the step-001 canvas):

```bash
uv run infer-fpn --image real_floorplan/BE_01_001.jpg --skip-step-2
```

Override output directory or canvas size via `run_fpn_pipeline` in code (`canvas_size`, `output_dir`). Default canvas uses `fpn.size` from config (768 px).

Outputs land in `output/fpn/<stem>/`:

| Step | Contents |
|---|---|
| **step-001** | Floorplan region mask/json, masked input, FPN masks, wall centerlines, fitted polygon JSON |
| **step-002** | Cropped re-infer (skipped with `--skip-step-2`) |
| **step-003** | LLM-refined room/door/window polygons, `final-masks/` PNG + JSON, `input.jpg`, `polygon-overlay.jpg` |

Step-003 `final-masks/` includes mask PNGs, polygon JSON, source `input.jpg`, and a translucent `polygon-overlay.jpg` preview.

## Manual labeling

Interactive pygame UI to refine structure masks. Source images live in `real_floorplan/`.

```bash
uv run label-data
# or
uv run python label_data.py
```

### What gets saved

| Location | Contents |
|---|---|
| `floorplan-manual-mask/input/<id>.jpg` | Source fit to 1024×1024 (unmasked) |
| `floorplan-manual-mask/output/<id>/` | Mask PNGs (white = active) + `polygon-overlay.jpg` |
| `output/label_data/<id>/step-003/final-masks/` | Polygon JSON edited in place + preview images |

Wall is mask-only (brush or **Compute wall**). Room, door, window, and background use polygon edit. Masks are derived from polygons on save.

Label-data inference runs the FPN pipeline at **1024 px** with **step-002 skipped**, so viewer coordinates match step-003 output.

### Batch auto-labeling

Run processed inference for every image in `real_floorplan/` (no UI):

```bash
uv run python label_data.py --auto
```

Skips images that already have complete outputs (all mask PNGs in `floorplan-manual-mask/output/<id>/` plus `output/label_data/<id>/step-003/final-masks/room.json`). Re-run everything with:

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
