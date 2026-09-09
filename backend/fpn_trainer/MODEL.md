# StarRoomFPN — model knowledge

Star-prompted room segmentation: a red star marks one room on the input; the network predicts that room's boundary, doors, and windows.

See `uv run fpn-train inspect` for live architecture and parameter counts from the active `config.yaml`.

---

## End-to-end flow

```
Hub image (1024, white bg + black walls)
    │
    ▼  prepare_image()
grayscale → resize 512 → invert → RGB (black bg + white walls)
    │
    ▼  draw_filled_star()
RGB + red star at random point inside target room
    │
    ▼  StarRoomFPN
mask logits (3×512×512) → sigmoid → 3-channel structure mask
    R = room boundary (Canny)
    G = door lines for this room
    B = window lines for this room
```

**Ground truth:** `build_room_targets()` → `(3, H, W)`. Ch0 = Canny boundary of wall-carved interior. Ch1/ch2 = line renders of doors/windows where `room.name` is in `connects` / `room_names`. Star placement uses the filled interior. Viz combines channels as RGB via `masks_to_rgb()`.

**Prompt:** the star replaces an invisible point prompt (similar idea to `sam_trainer`, but visible on the image).

---

## Module reference

### `star.py` — star prompt

Draws a filled 5-point star at `(cx, cy)` with outer radius `r`. Vertices alternate outer/inner radius (`inner_ratio`, default 0.4). During training, position and size are randomized; the star must fit inside the room mask.

### `data.py` — dataset

| Piece | Role |
|---|---|
| `prepare_image()` | Grayscale → resize → invert → RGB |
| `scale_extraction()` | Scale room/wall polygon coords to `input_size` |
| `rasterize_room_mask()` | Fill room polygon, subtract wall strips (star sampling) |
| `room_boundary_mask()` | Canny edges of interior fill → ch0 |
| `rasterize_line_mask()` | Door/window line segments → ch1/ch2 |
| `build_room_targets()` | Stack boundary + doors + windows → `(3, H, W)` |
| `sample_point_from_mask()` | Random star center, weighted toward room interior |
| `FloorplanStarDataset` | Hub rows → `(starred_rgb, targets)` pairs |
| `load_train_split()` | Cached Hub dataset (no streaming) |

Each floor plan yields `room_slots × per_room_variation` samples per epoch (default 4).

### `net/blocks.py` — building blocks

| Class | Role |
|---|---|
| `ConvBnSiLU` | Conv2d + GroupNorm + SiLU |
| `ResidualBlock` | Two 3×3 conv blocks with skip connection |

### `net/backbone.py` — `CnnBackbone`

4-stage CNN encoder. Stem stride-2, then four `DownStage`s (stride-2 conv + residual blocks).

Outputs **c2…c5** at strides 4, 8, 16, 32 (for `input_size=512`: 128², 64², 32², 16²).

| Level | Typical role |
|---|---|
| c2 | Wall edges, star, local detail |
| c3 | Room-scale shapes |
| c4 | Layout chunks |
| c5 | Whole-plan summary (heavily pooled) |

### `net/fpn.py` — `YoloLikeFPN`

Multi-scale neck with **lateral** + **top-down** only (bottom-up removed for now).

#### Lateral

1×1 conv on each backbone level → unified channel width (`fpn_width`). Puts c2…c5 in the same feature space before merging.

```
c2 (128²) ──► l2 (128²)
c3 (64²)  ──► l3 (64²)
c4 (32²)  ──► l4 (32²)
c5 (16²)  ──► l5 (16²)
```

#### L5 transformer (global reasoning)

After lateral projection, **l5** passes through `L5Transformer` before top-down fusion. This adds explicit all-pairs self-attention on the coarsest FPN level (256 tokens at 512 input).

```
l5 (16², 64 ch)
    │
    ▼  L5Transformer (3D RoPE, depth=2)
l5′ (16², 64 ch)   ← global context mixed, residual add
    │
    ▼
P5 = l5′
```

**Why L5?** Smallest grid (cheap attention), each cell covers ~32×32 px — the whole floor plan is compressed into 16×16 tokens.

#### Top-down

Coarse features flow down to fine levels. Each fine pixel receives compressed global layout context (now including transformer-refined L5).

```
P5 = l5′
P4 = merge(l4 + upsample(P5))
P3 = merge(l3 + upsample(P4))
P2 = merge(l2 + upsample(P3))   ← used by mask head (with P3, P4)
```

**Global context:** L5 transformer provides explicit attention; top-down FPN propagates that context to finer scales.

### `net/l5_transformer.py` — `L5Transformer`

Self-attention on the L5 feature map with **axial 3D RoPE** over `(y, x, pyramid level)`.

| Component | Role |
|---|---|
| `Rotary3D` | Precomputed cis table; splits head_dim into thirds for y, x, level |
| `TransformerBlock` | Pre-norm attention + MLP (same pattern as `unet_trainer`) |
| `L5Transformer` | 1×1 in/out proj (when `dim ≠ fpn_width`), flatten → blocks → reshape, residual add |

**3D RoPE:** extends the 2D axial RoPE in `unet_trainer` with a third axis for pyramid level. At L5 all tokens share the same `level` index (default `4`), which tags this grid as the coarsest FPN scale. Requires `head_dim % 6 == 0` (each axis gets an even slice for complex rotation).

Config (`config.yaml`):

```yaml
net:
  l5_transformer:
    depth: 2
    dim: 48          # internal transformer width (64→48→64 via 1×1 when fpn_width=64)
    heads: 4         # head_dim=12, divisible by 6
    mlp_ratio: 4.0
    level: 4         # RoPE z-axis: L5 pyramid index
    max_l5_grid_side: 64  # max L5 side length for 3D RoPE (input_size / 32 ≤ this)
```

Remove the `l5_transformer` block (or set to null) to disable and revert to CNN-only global context.

### `net/mask_head.py` — `FPNMaskHead`

Fuses **P2 + P3 + P4** (upsampled to P2 resolution), 3×3 convs, 1×1 → `out_channels` (default 3), upsample 4× to full `input_size`.

P2 = finest detail + top-down context; P3/P4 = mid/coarse scales.

### `net/loss.py` — mask loss

Per-channel dice + focal, summed (`multi_mask_loss`). Optional `loss.channel_weights`. No box or classification loss.

### `canvas.py` — viz

`masks_to_rgb()`: R=boundary, G=doors, B=windows. `overlay_masks()` blends over the starred input.

### `net/model.py` — `StarRoomFPN`

Composes backbone → FPN (with optional L5 transformer) → mask head.

---

## Model sizes

Two configs were explored. **Active default** includes the L5 transformer.

### 0.8M + L5 transformer (current default)

```yaml
net:
  stem_channels: 16
  backbone_channels: [32, 48, 64, 64]
  depth: 1
  fpn_width: 64
  l5_transformer:
    depth: 2
    dim: 48
    heads: 4
    mlp_ratio: 4.0
    level: 4
    max_l5_grid_side: 64
```

| | |
|---|---|
| Total params | **628,705** (~629K) |
| `input_size` | 512 |
| Use | Fast iteration with explicit global reasoning on L5 |

Per-layer (current default):

| Layer | Params |
|---|---|
| backbone.stem | ~480 |
| backbone.stages.0–3 | ~23K / ~56K / ~102K / ~111K |
| fpn.lateral | ~14K |
| fpn.l5_transformer | ~63K |
| fpn.merge_td | ~111K |
| mask_head | ~148K |

### 3.6M (previous / larger variant, no transformer)

```yaml
net:
  stem_channels: 32
  backbone_channels: [48, 96, 128, 128]
  depth: 2
  fpn_width: 128
```

| | |
|---|---|
| Total params | **~2.67M** (no bottom-up; no L5 transformer) |
| Use | More CNN capacity if 0.8M underfits walls or large rooms |

Checkpoints are **not interchangeable** between size configs or when toggling the L5 transformer.

---

## Bottom-up (removed, documented for later)

Previously the FPN also ran a **bottom-up** path (PAN-style):

```
P2 (128²) ──down──► N3 = merge(P3 + down(P2))
N3 ──down──► N4 = merge(P4 + down(N3))
N4 ──down──► N5 = merge(P5 + down(N4))
```

**Purpose:** after top-down injects global context, bottom-up re-injects fine detail so walls stay sharp.

**Why removed:** N3–N5 were computed but never fed to the mask head (~222K unused params in the 0.8M config; ~887K in the 3.6M config). Top-down alone is sufficient to start training.

**How to re-enable later:**

1. Restore `down` + `merge_bu` in `net/fpn.py`.
2. Wire into the mask head — e.g. `head(p2, n3, p4)` or concat `[p2, n3, p3, n4]`.
3. See options in project chat: swap P→N, concat P+N, or add N2 at 128² for full PAN.

---

## Spatial shapes (`input_size=512`)

| Tensor | Shape |
|---|---|
| Input | (3, 512, 512) |
| c2 / P2 | (C, 128, 128) |
| c3 / P3 | (C, 64, 64) |
| c4 / P4 | (C, 32, 32) |
| c5 / l5 / P5 | (C, 16, 16) |
| L5 tokens | (256, dim) before reshape |
| mask logits | (512, 512) |

---

## What the model can and cannot do

**Can:**

- Segment the room containing the star using local walls + coarse layout (via L5 attention + top-down FPN).
- Reason globally over the whole floor plan at L5 (256-token self-attention).
- Handle varied star size/position (training augmentation).

**Cannot (without further changes):**

- All-pairs attention at full 512² resolution (only L5 is transformed).
- Pick a room without the star prompt.
- Match ViT/SAM quality on thin walls with the 0.8M CNN backbone alone.

---

## File map

```
net/
├── blocks.py         ConvBnSiLU, ResidualBlock
├── backbone.py       CnnBackbone → [c2,c3,c4,c5]
├── fpn.py            YoloLikeFPN (lateral + L5 transformer + top-down)
├── l5_transformer.py L5Transformer, Rotary3D (y,x,level)
├── mask_head.py      FPNMaskHead → mask logits
├── loss.py           dice + focal
└── model.py          StarRoomFPN, build_model()
```
