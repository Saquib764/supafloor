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
| `image` | image | Realistic 1024×1024 floor-plan JPG |
| `extraction` | string | Full extraction JSON for the same image |
| `simple_image` | image | Simple grayscale render (black walls, gray-80 windows, gray-160 doors) |

## Local layout (before / after Hub)

```
input/<stem>.jpg
output/<stem>.json
output2/<stem>.jpg   # simple_image; uploaded to Hub
stats/<stem>.yml     # local export only (transform stats; not uploaded to Hub)
```

## Extraction JSON

Each `output/<stem>.json` file matches the rendered image. Room polygons are on the 1024×1024 canvas and are canonicalized after rotation and scaling:

- **Winding:** clockwise
- **Start vertex:** closest to the origin `(0, 0)`

The Florence trainer reads these polygons and encodes loc-token labels on the fly.

Wall rendering metadata (plan-level):

| Field | Type | Meaning |
|---|---|---|
| `wall_render_mode` | `1 \| 2 \| 3` | `1` all walls textured, `2` mixed, `3` no texture |
| `wall_hatch` | string \| null | Matplotlib hatch for textured walls; `null` = solid fill |
| `wall_texture_cutoff` | float | Fraction textured in mixed mode |
| `wall_centerline` | bool | Thin axis line on every wall (independent of texture) |

Every wall is drawn with an outline (strip edge). Texture fill is optional per wall via `is_textured`. Each wall segment also includes `is_exterior` (bool).

Room interior texture (per room):

| Field | Type | Meaning |
|---|---|---|
| `is_interior_textured` | bool | ~10% chance per room |
| `interior_hatch` | string | `"."` or `"/"` when textured |

## Dataset size (basic_5000)

| Metric | Value |
|---|---:|
| Plans | 5000 |
| Variants | 25000 |
| Variants per plan | 5 |

Augmentations intentionally include rotation, missing room labels, missing door swing symbols, and occasional pillars so models see messy scans.
