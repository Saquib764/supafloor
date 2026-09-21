# floorplan-manual-mask

Manually corrected five-class structure masks for FPN Five training.

## Local layout

```
floorplan-manual-mask/
  input/<stem>.jpg
  output/<stem>/background.png
  output/<stem>/room.png
  output/<stem>/wall.png
  output/<stem>/door.png
  output/<stem>/window.png
  output/<stem>/polygon-overlay.jpg
```

White pixels are active in each mask PNG. `polygon-overlay.jpg` is a translucent preview of room/door/window/boundary polygons on the input image.

Annotate locally with `backend/inference_core/label_data.py`. Polygon JSON is stored separately under `backend/inference_core/output/label_data/<stem>/step-003/final-masks/`.

Batch inference (skips images that already have masks + polygons):

```bash
cd backend/inference_core
uv sync --group dev
uv run python label_data.py --auto
```

## Training

Set in `backend/fpn_five_trainer/config.yaml`:

```yaml
data:
  dataset: floorplan-manual-mask
  source: local
```

With `source: auto`, the trainer uses the local folder when `floorplan-manual-mask/input/` exists, otherwise the Hub.
