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

- **Prompt:** `Locate room polygon containing point <loc_x><loc_y>[, and not containing <loc_x><loc_y>]`
- **Label:** `<poly>…</poly>` + `Room name: {name|unlabeled}`

Loc-token labels are derived from `extraction.rooms[].polygon` at train time (not precomputed on export). Polygons in extraction JSON are clockwise with the first vertex closest to the origin.
