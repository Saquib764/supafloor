# Florence rooms (training view)

Hub view over [`saquiboye/floorplan-to-json`](https://huggingface.co/datasets/saquiboye/floorplan-to-json) for Florence-2 LoRA referring-expression segmentation.

Loads only the columns the trainer needs:

| Field | Type | Meaning |
|---|---|---|
| `stem` | string | `{plan_id}_{variant}` |
| `plan_id` | string | Unique apartment / plan id (train/val split key) |
| `variant` | int | Augmentation index |
| `image` | image | Floor-plan PNG |
| `extraction` | string | Extraction JSON (rooms + polygons on 1024 canvas) |

At training time the Florence trainer picks a random room from `extraction`, samples a random interior point, and builds:

- **Prompt:** `<REFERRING_EXPRESSION_SEGMENTATION>room containing <loc_x><loc_y>`
- **Label:** `<poly>…</poly>` + `Room name: {name|unlabeled}`

Precomputed `florence` labels from the full Hub dataset are not used.
