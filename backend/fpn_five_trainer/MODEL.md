# FloorplanFPN — five-class model

Full floor-plan structure segmentation: grayscale schematic in, five mask channels out (background, walls, rooms, doors, windows).

See `uv run fpn-train inspect` for live architecture and parameter counts from the active `config.yaml`.

---

## End-to-end flow

```
Hub image (1024, white bg + black walls)
    │
    ▼  prepare_image()
grayscale → resize → invert (black bg + white walls), single channel
    │
    ▼  FloorplanFPN
mask logits (5×H×W) → sigmoid → 5-channel structure mask
    ch0 = background (exterior / non-structure)
    ch1 = walls
    ch2 = room interiors
    ch3 = doors
    ch4 = windows
```

**Ground truth:** `build_floorplan_targets()` → `(5, H, W)`.

| Channel | Rasterizer |
|---|---|
| 0 background | `1 - clip(wall + room)` |
| 1 walls | `rasterize_wall_mask` (uncut edges + junction extend + pillars) |
| 2 rooms | `rasterize_all_rooms_mask` (interiors, walls carved out) |
| 3 doors | `rasterize_opening_mask(doors)` |
| 4 windows | `rasterize_opening_mask(windows)` |

Channels overlap: doors/windows sit on wall geometry. Background is the complement of walls ∪ rooms (exterior/outside footprint).

---

## Loss

`five_mask_loss`: per-channel dice + focal, weighted by `loss.channel_weights`.

- **Walls (ch1):** extra false-negative penalty in door∪window regions (from `fpn_trainer`).
- **Rooms (ch2):** extra false-positive penalty in door∪window regions (from `fpn_room_trainer`).

---

## Module reference

Same backbone/FPN/L5 stack as `fpn_trainer`, extended with:

- **P1 @ stride 2** — stem lateral + top-down from P2; mask head decodes from P1 with 2× upsample.
- **DW 7×7 context** on P3 and P4 after top-down merge (large receptive field for long openings).

Mask head output width: `out_channels: 5`.

```
net/
├── blocks.py         ConvBnSiLU, ResidualBlock
├── backbone.py       CnnBackbone → [c2,c3,c4,c5]
├── fpn.py            YoloLikeFPN (lateral + L5 transformer + top-down)
├── l5_transformer.py L5Transformer, Rotary2D (y,x) with i/(L-1) normalized coords
├── mask_head.py      FPNMaskHead → 5 mask logits
├── loss.py           five_mask_loss
└── model.py          FloorplanFPN, build_model()
```

---

## Spatial shapes (`input_size=512`)

| Tensor | Shape |
|---|---|
| Input | (1, 512, 512) |
| mask logits | (5, 512, 512) |

Checkpoints are **not interchangeable** with `fpn_trainer` (3 ch) or `fpn_room_trainer` (1 ch).
