# Inference core

SAM mask-fit room polygon extraction for floor plans.

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

## Config

| Key | Meaning |
|---|---|
| `sam_config` | Path to `sam_trainer/config.yaml` (checkpoint, model_id) |
| `grid.size` | Grid cells per side (default 100) |
| `point_filter.dilate_px` | Wall dilation radius (px) for point filter image (invert → dilate → invert) |
| `mask_smooth.close_px` | Morphological close radius (px) before smoothing |
| `mask_smooth.smooth_sigma` | Gaussian blur sigma for boundary smoothing |
| `mask_smooth.approx_epsilon` | Douglas–Peucker epsilon when simplifying the mask boundary |
| `fit.min_sides` | Minimum polygon sides when fitting a mask |
| `fit.max_sides` | Maximum polygon sides when fitting a mask |
| `fit.output_dir` | Directory for fit overlays |
