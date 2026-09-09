# Inference core

SAM mask-fit room polygon extraction for floor plans, with optional Replicate image-edit cleaning.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.11. Uses weights from `backend/sam_trainer/`.

```bash
cd backend/inference_core
uv sync
```

## Mask-fit loop

Samples a point, runs SAM, fits simplified polygons across the configured side range, snaps to corners, refines against Sobel edges, saves an overlay, cleans the working image, and repeats.

```bash
uv run infer-sam-fit --image test_image/fp01.jpg
```

Press **Enter** to continue after each step, **Esc** to exit.

Outputs land in `output/fit/` as `step-NNN.png` — best-fit polygon and SAM mask only. Sobel and corner debug images are written to `output/`.

## Image edit + room mask discovery

Fits the input to a 1024×1024 white canvas, runs three chained Replicate image-edit steps (structure clean → close openings → remove furniture), saves each step, then finds room masks as connected white regions in the grayscaled final image.

```bash
uv run infer-image-edit --image test_image/fp01.jpg
```

Use `--skip-replicate` to reuse saved `03-no-furniture.png`.

Outputs land in `output/image_edit/<stem>/` (`00-canvas.png` … `03-no-furniture.png`, plus `*-diff.png` per step) with room masks in `masks/gray.png`, `masks/step-NNN.png`, and `masks/all-masks.png`.

Set `REPLICATE_API_TOKEN` or `replicate_token` in `config.yaml`.

## Config

| Key | Meaning |
|---|---|
| `replicate_token` | Replicate API token (overridden by `REPLICATE_API_TOKEN`) |
| `image_edit.model` | Replicate image-edit model id (default `openai/gpt-image-2.5-flare`) |
| `image_edit.quality` | GPT Image quality (`medium`, `high`, etc.) |
| `image_edit.aspect_ratio` | Output aspect ratio (`1:1` for 1024 canvas) |
| `image_edit.output_format` | Output format (`png`, `webp`, `jpeg`) |
| `image_edit.output_dir` | Base directory for per-image cleaning outputs |
| `image_edit.prompts.*` | Prompts for the cleaning steps |
| `room_masks.white_threshold` | Grayscale value treated as white (default 250) |
| `room_masks.min_area_px` | Minimum connected white region area to keep |
| `room_masks.exclude_largest` | Drop largest white component (canvas background) |
| `sam_config` | Path to `sam_trainer/config.yaml` (checkpoint, model_id) |
| `grid.size` | Grid cells per side (default 100) |
| `point_filter.dilate_px` | Wall dilation radius (px) for point filter image (invert → dilate → invert) |
| `mask_smooth.close_px` | Morphological close radius (px) before smoothing |
| `mask_smooth.smooth_sigma` | Gaussian blur sigma for boundary smoothing |
| `mask_smooth.approx_epsilon` | Douglas–Peucker epsilon when simplifying the mask boundary |
| `fit.min_sides` | Minimum polygon sides when fitting a mask |
| `fit.max_sides` | Maximum polygon sides when fitting a mask |
| `fit.output_dir` | Directory for fit overlays |
