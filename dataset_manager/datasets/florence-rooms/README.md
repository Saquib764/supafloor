# Florence rooms (training view)

Hub view over [`saquiboye/floorplan-to-json`](https://huggingface.co/datasets/saquiboye/floorplan-to-json) for Florence-2 LoRA training.

Loads only the columns the trainer needs:

| Field | Type | Meaning |
|---|---|---|
| `stem` | string | `{plan_id}_{variant}` |
| `plan_id` | string | Unique apartment / plan id (train/val split key) |
| `variant` | int | Augmentation index |
| `image` | image | Floor-plan PNG |
| `florence` | string | Loc-token label |

`extraction` from the full Hub dataset is dropped at load time in `schema.py`. Publish data via `floorplan-to-json`.
