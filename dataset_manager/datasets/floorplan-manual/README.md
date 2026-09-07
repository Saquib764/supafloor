---
license: other
task_categories:
  - image-to-text
language:
  - en
tags:
  - floorplan
  - florence-2
  - room-polygons
  - manual
pretty_name: Floorplan Manual
size_categories:
  - n<1K
---

# Floorplan Manual

Manually annotated residential floor-plan images paired with room-polygon extraction JSON and simple renders.

Same local folder layout as [`saquiboye/floorplan-to-json`](https://huggingface.co/datasets/saquiboye/floorplan-to-json). Hub columns match the synthetic dataset except each manual sample is a single row with `variant: 0`.

| Field | Type | Meaning |
|---|---|---|
| `stem` | string | Image basename (e.g. `001`, `fp01`) |
| `plan_id` | string | Random 12-char hex id assigned at upload |
| `variant` | int | Always `0` |
| `image` | image | Original floor-plan JPG |
| `extraction` | string | Room-polygon JSON (`id`, `rooms[]`) |
| `simple_image` | image | Simple grayscale render |

## Local layout (before / after Hub)

```
input/<stem>.jpg
output/<stem>.json
output2/<stem>.jpg
```

Annotated locally with `backend/floorplan/dev/annotate_rooms.py` into `floorplan-manual/` at the repo root.

## Florence training

Set `data.dataset: floorplan-manual` in `backend/florence_trainer/config.yaml`. The trainer reads `image` and `extraction`; loc-token labels are built at train time.
