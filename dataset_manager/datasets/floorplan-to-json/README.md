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
  - synthetic
pretty_name: Floorplan to JSON
size_categories:
  - 10K<n<100K
---

# Floorplan to JSON

Synthetic residential floor-plan images paired with extraction JSON (rooms, walls, openings, and render metadata).

Each row is one augmented render of a unique apartment layout. Florence loc-token training labels are **not** stored here — the trainer builds them from `extraction` at train time.

| Field | Type | Meaning |
|---|---|---|
| `stem` | string | `{plan_id}_{variant}` |
| `plan_id` | string | Unique apartment / plan id |
| `variant` | int | Augmentation index (`0`–`4`) |
| `image` | image | Realistic 1024×1024 floor-plan PNG |
| `extraction` | string | Full extraction JSON for the same image |

## Local layout (before / after Hub)

```
input/<stem>.png
output/<stem>.json
stats/<stem>.yml     # local export only (transform stats; not uploaded to Hub)
```

## Extraction JSON

Each `output/<stem>.json` file matches the rendered image. Room polygons are on the 1024×1024 canvas and are canonicalized after rotation and scaling:

- **Winding:** clockwise
- **Start vertex:** closest to the origin `(0, 0)`

The Florence trainer reads these polygons and encodes loc-token labels on the fly.

## Dataset size (basic_5000)

| Metric | Value |
|---|---:|
| Plans | 5000 |
| Variants | 25000 |
| Variants per plan | 5 |

Augmentations intentionally include rotation, missing room labels, missing door swing symbols, and occasional pillars so models see messy scans.
