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
```

White pixels are active in each mask PNG. Annotate locally with `backend/inference_core/label_data.py`.

## Training

Set in `backend/fpn_five_trainer/config.yaml`:

```yaml
data:
  dataset: floorplan-manual-mask
  source: local
```

With `source: auto`, the trainer uses the local folder when `floorplan-manual-mask/input/` exists, otherwise the Hub.
