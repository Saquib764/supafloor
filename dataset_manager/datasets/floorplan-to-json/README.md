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

Synthetic residential floor-plan images paired with Florence-2 loc-token labels and extraction JSON.

Each row is one augmented render of a unique apartment layout.

| Field | Type | Meaning |
|---|---|---|
| `stem` | string | `{plan_id}_{variant}` |
| `plan_id` | string | Unique apartment / plan id |
| `variant` | int | Augmentation index (`0`–`4`) |
| `image` | image | Realistic 1024×1024 floor-plan PNG |
| `florence` | string | Florence loc-token target (room names + polygons) |
| `extraction` | string | Full extraction JSON for the same image |

## Local layout (before / after Hub)

```
input/<stem>.png
florence/<stem>.txt
output/<stem>.json
```

## Label format

Each room in `florence`:

```
Living
<loc_158><loc_570>
<poly>
<loc_157><loc_569>
<loc_463><loc_569>
</poly>
<sep>
```

Name, center loc pair, then polygon vertices. Loc bins are `0`–`999`.

## Dataset size (basic_5000)

| Metric | Value |
|---|---:|
| Plans | 5000 |
| Variants | 25000 |
| Variants per plan | 5 |

Augmentations intentionally include rotation, missing room labels, missing door swing symbols, and occasional pillars so models see messy scans.
