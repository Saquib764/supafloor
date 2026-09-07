# Florence-2 architecture and LoRA review

Reviewed: 2026-09-07

## Scope

This document consolidates the architecture and LoRA investigations performed for
the floor-plan room-polygon trainer. It covers:

- the published Florence-2 architecture;
- the exact Microsoft remote-code implementation loaded by this repository;
- what each important module does;
- where LoRA can be placed and the expected effect of each placement;
- the repository's current trainable/frozen parameter plan;
- a graded review of that plan, including strengths, weaknesses, and recommended
  ablations.

Statements marked **verified** come directly from the Florence-2 paper, upstream
configuration/source, PEFT behavior, or this repository. Statements marked
**expected effect** are engineering hypotheses inferred from information flow.
Transformer modules are strongly coupled, so no individual layer exclusively
"controls" a semantic property. Placement claims must ultimately be confirmed by
controlled ablations on this dataset.

## Executive assessment

**Overall grade: B+ (8.3/10).**

The current setup is substantially better than a generic `q_proj`/`v_proj` LoRA:
it adapts the vision tower where region structure becomes semantic, adapts every
language encoder attention layer, gives the decoder more capacity, includes all
decoder cross-attention paths, and fully trains the vision-to-language projector.
It is carefully validated against accidental trainable modules.

Its main weakness is not an obviously wrong target choice. The larger weakness is
the lack of validation-driven evidence for the choices. Training uses no
evaluation split, no geometry metric, and no target/rank ablation, so the current
selection cannot be shown to be optimal. The design also gives the same rank to
decoder self-attention and cross-attention even though cross-attention is more
directly responsible for image-grounded coordinates, excludes all FFNs, and saves
the fully trained image projection outside the standard PEFT adapter artifact.

Recommended immediate direction:

1. Keep the current configuration as the baseline.
2. Add a held-out evaluation protocol before changing targets.
3. Compare decoder cross-attention rank 16 with decoder self-attention rank 8.
4. If output syntax is valid but shapes are inaccurate, test selected vision FFNs
   or additional stage-2 spatial-attention blocks.
5. If output syntax, vertex order, closure, or token choice is poor, test decoder
   FFNs before broadening the vision tower.
6. Continue using explicit fully qualified target paths. Do not replace them with
   `target_modules="all-linear"` for this remote-code model.

## 1. Florence-2 in one sentence

Florence-2 is a prompt-conditioned image-to-sequence model: a DaViT image encoder
turns an image into visual tokens, a BART-style encoder jointly contextualizes
those visual tokens and prompt tokens, and an autoregressive decoder emits text
and quantized location tokens through one shared language-modeling head.

Unlike a conventional detector or segmenter, Florence-2 does not use a separate
box regression, mask, or polygon head. Coordinates are vocabulary tokens, and
geometry is generated as a sequence.

## 2. End-to-end data flow

For the configured `microsoft/Florence-2-base` checkpoint and a 768×768 input:

```text
RGB image [B, 3, 768, 768]
  │
  ├─ DaViT stage 0: stride 4  → [B, 192×192, 128]
  ├─ DaViT stage 1: stride 2  → [B,  96×96, 256]
  ├─ DaViT stage 2: stride 2  → [B,  48×48, 512]
  └─ DaViT stage 3: stride 2  → [B,  24×24, 1024]
                                      │
                                      ├─ 576 spatial tokens
                                      ├─ learned 2-D image positions
                                      ├─ temporal position (T=1 here)
                                      └─ one global pooled token
                                           ↓
                                 [B, 577, 1024]
                                           │
                         image_projection [1024, 768]
                         image_proj_norm   LayerNorm(768)
                                           ↓
                                 [B, 577, 768]
                                           │
Prompt text → token embedding [B, Nt, 768]─┘
                                           ↓ concatenate
                                 [B, 577+Nt, 768]
                                           │
                          6-layer multimodal encoder
                                           │
                          6-layer autoregressive decoder
                          ├─ causal target self-attention
                          ├─ cross-attention to encoder
                          └─ tokenwise FFN
                                           │
                              tied LM head → 51,289 logits
                                           │
                        `<poly><loc_x><loc_y>...</poly>`
```

The combined stride is 32, so each final 24×24 token is nominally aligned to a
32×32 input region. Its actual receptive field is larger because prior
convolutions and attention blocks have already mixed information.

The upstream remote code names two feature sources:

- `temporal_avg_pool`: with one image (`T=1`), averaging over time preserves all
  576 spatial positions;
- `spatial_avg_pool`: averaging over the 576 positions contributes one global
  image token.

Concatenating them produces 577 visual tokens. The names can be misleading for a
single-image model: `temporal_avg_pool` is the spatial token grid, not one pooled
temporal token.

## 3. Base-model dimensions

### 3.1 Whole model

- Published size: 232 million parameters.
- Inspected checkpoint size: 231,414,016 parameters.
- Vocabulary: 51,289 tokens.
- Coordinate vocabulary: 1,000 location bins.
- Pretraining objective: autoregressive token cross-entropy over text and
  location-token responses.
- Upstream high-resolution tuning resolution: 768×768.

### 3.2 DaViT vision tower

- Stage dimensions: `[128, 256, 512, 1024]`.
- Stage depths: `[1, 1, 9, 1]`.
- Spatial-attention heads: `[4, 8, 16, 32]`.
- Channel-attention groups: `[4, 8, 16, 32]`.
- Patch strides: `[4, 2, 2, 2]`.
- Spatial window size: 12.
- Approximate vision-backbone parameters reported by the paper: 90 million.

### 3.3 Language encoder-decoder

- Architecture: BART-style encoder-decoder.
- Hidden size: 768.
- Encoder: 6 layers, 12 attention heads per layer.
- Decoder: 6 layers, 12 heads in self-attention and cross-attention.
- Encoder and decoder FFN width: 3,072.
- Maximum position embeddings: 1,024.
- Approximate language-component parameters reported by the paper: 140 million.

## 4. Vision tower, module by module

### 4.1 Patch/downsampling convolutions

Remote-code path:

```text
vision_tower.convs.{stage}.proj
```

Each stage begins with a convolutional embedding/downsampling operation. Stage 0
converts RGB pixels to 128-dimensional visual features; later stages lower the
spatial resolution and raise channel width.

**Verified role:** local pixel aggregation, downsampling, and channel projection.

**Expected effect if adapted:**

- early stages: stroke thickness, contrast, scan/render style, line and text
  appearance, very local corners and edges;
- later stages: how lower-resolution structures are formed before attention.

These modules are expensive and risky to alter broadly because they define the
base visual representation. They are reasonable targets when the new image
domain is visually very different from pretraining, but less compelling when the
base tower already detects floor-plan lines reliably.

### 4.2 Depthwise convolutions inside DaViT blocks

Each spatial and channel block contains depthwise 3×3 convolutions around
attention/FFN processing.

**Verified role:** inject local spatial mixing while preserving separate channels.

**Expected effect if adapted:** local edge continuity, junction smoothing, and
small-neighborhood texture handling. Ordinary linear-only LoRA does not target
these `Conv2d` modules. PEFT can LoRA-wrap supported convolutions when explicitly
configured, but doing so should be treated as a separate experiment.

### 4.3 Spatial window attention

Remote-code paths:

```text
vision_tower.blocks.{stage}.{block}.spatial_block.window_attn.fn.qkv
vision_tower.blocks.{stage}.{block}.spatial_block.window_attn.fn.proj
```

The input token grid is partitioned into non-overlapping 12×12 windows. Attention
is performed among spatial positions within each window.

`qkv` is one fused `Linear(D, 3D)` module. The result is split into query, key,
and value tensors only after projection. A standard PEFT LoRA around this module
therefore updates the full fused operation; it is not independently masking only
Q or V.

**Verified role:** local spatial relationships and local feature aggregation.

**Expected effect if adapted:** wall/corner grouping, nearby boundary continuity,
local room shape, short-range relationships, and separating adjacent structures.

At 768 input resolution:

- stage 2 has a 48×48 grid; a 12×12 window spans a nominal 192×192 input area;
- stage 3 has a 24×24 grid; a window spans a nominal 384×384 input area.

The true context is larger because channel attention carries global information
and earlier blocks have already mixed features.

### 4.4 Channel-group attention

Remote-code paths:

```text
vision_tower.blocks.{stage}.{block}.channel_block.channel_attn.fn.qkv
vision_tower.blocks.{stage}.{block}.channel_block.channel_attn.fn.proj
```

DaViT transposes the usual attention perspective: channels become tokens whose
features summarize spatial positions. Channels are grouped for efficiency, and
attention operates within each group.

**Verified role:** global image context through interactions among feature
channels while retaining linear spatial complexity.

**Expected effect if adapted:** selecting floor-plan-relevant feature families,
distinguishing walls from furniture/text/noise, global structural consistency,
and suppressing visual features that are useful for natural images but distracting
for architectural drawings.

Spatial and channel attention are complementary:

- spatial attention refines local position relationships;
- channel attention provides global context and changes which feature channels
  matter.

### 4.5 Vision attention output projection (`proj`)

After attention combines values, `proj` maps the result back into the block's
residual feature space.

**Verified role:** remix attention heads/groups and determine how retrieved
information is written back to the residual stream.

**Expected effect if adapted:** task-specific rewriting of attended evidence. It
can be a high-value target because it changes the usable representation without
directly changing the Q/K matching function.

### 4.6 Vision FFNs (`fc1`, `fc2`)

Remote-code paths:

```text
...spatial_block.ffn.fn.net.fc1  # D → 4D
...spatial_block.ffn.fn.net.fc2  # 4D → D
...channel_block.ffn.fn.net.fc1  # D → 4D
...channel_block.ffn.fn.net.fc2  # 4D → D
```

`fc1` expands each token into a wider nonlinear feature space; GELU activates it;
`fc2` contracts the result back into the residual width.

**Verified role:** tokenwise nonlinear feature transformation.

**Expected effect if adapted:**

- `fc1`: create or select task-specific intermediate features;
- `fc2`: decide how those features are written into the residual representation.

FFN adaptation offers broader representational change than attention-only LoRA.
That may help if attention finds the right area but features are insufficient to
describe exact floor-plan geometry. It also costs more parameters and can more
aggressively alter pretrained visual semantics.

### 4.7 Learned image-position embedding

Remote-code path:

```text
image_pos_embed
```

Florence-2 adds a learned absolute 2-D position embedding to the final visual
grid before projection.

**Verified role:** provide absolute row/column position.

**Expected effect if adapted:** absolute coordinate calibration and position
bias. This is relevant to location-token prediction, but changing it risks
distorting a coordinate system that was already high-resolution-tuned at 768.
It is not a normal LoRA target and is currently frozen.

### 4.8 Temporal position embedding

Remote-code path:

```text
visual_temporal_embed
```

The implementation supports a temporal dimension, but this trainer passes one
image, so `T=1`.

**Expected value for this task:** very low. There is no video-time relationship
to learn.

## 5. Multimodal bridge

### 5.1 `image_projection`

Remote implementation:

```text
image_projection  # raw nn.Parameter [1024, 768]
```

This matrix maps final DaViT features from width 1024 to language width 768.
Because it is a raw `nn.Parameter`, not an `nn.Linear`, regular PEFT
`target_modules` does not wrap it in this implementation. The repository instead
fully trains all 786,432 entries.

**Verified role:** dimensional and representational alignment between vision and
language.

**Expected effect if adapted:** make floor-plan visual features legible to the
language encoder. This is a high-leverage adaptation point because every visual
token passes through it.

**Limitation:** it is global and not spatially selective. A fully trained
projector can learn a strong domain alignment, but can also overfit or drift more
than a low-rank update. It currently shares the same optimizer learning rate as
the LoRA weights.

### 5.2 `image_proj_norm`

Remote-code path:

```text
image_proj_norm  # LayerNorm(768)
```

**Verified role:** normalize projected visual-token statistics before they are
concatenated with prompt embeddings.

The repository trains and saves this module in full through
`modules_to_save=["image_proj_norm"]`. It has only 1,536 trainable values
(768 scale and 768 bias), so the capacity cost is negligible.

**Expected effect if adapted:** calibrate the scale and offset of visual channels
relative to text channels. This is a sensible companion to a trainable projector.

## 6. Multimodal language encoder

Before the encoder, Florence-2 concatenates projected visual tokens and prompt
token embeddings. Encoder self-attention can therefore mix:

- image position with image position;
- prompt token with prompt token;
- image position with prompt token in either direction.

For each encoder layer `i`:

```text
language_model.model.encoder.layers.{i}.self_attn.q_proj
language_model.model.encoder.layers.{i}.self_attn.k_proj
language_model.model.encoder.layers.{i}.self_attn.v_proj
language_model.model.encoder.layers.{i}.self_attn.out_proj
language_model.model.encoder.layers.{i}.fc1
language_model.model.encoder.layers.{i}.fc2
```

### 6.1 Encoder `q_proj`

**Verified mechanics:** constructs what each visual or prompt token asks for.

**Expected effect:** alter which context a point token, instruction word, or image
position seeks. For this task, it can improve matching "containing point" and
"not containing" prompt conditions to relevant visual regions.

### 6.2 Encoder `k_proj`

**Verified mechanics:** constructs how every encoder token advertises itself for
attention matching.

**Expected effect:** make walls, room interiors, prompt conditions, and supplied
location tokens easier or harder for other tokens to select.

### 6.3 Encoder `v_proj`

**Verified mechanics:** constructs the content returned from an attended token.

**Expected effect:** determine what geometric, semantic, or prompt-condition
information is carried after a position has been selected.

### 6.4 Encoder `out_proj`

**Verified mechanics:** remixes attention-head outputs and writes the result into
the residual stream.

**Expected effect:** change how fused image/prompt evidence is represented for
subsequent encoder layers and the decoder.

### 6.5 Encoder FFN

The encoder FFN transforms each already-contextualized token independently.

**Expected effect:** create nonlinear joint representations such as "this room
contains the requested point but not the excluded point." It may help a difficult
new reasoning relation, but the current task can plausibly be learned through
attention routing plus the already capable pretrained FFNs.

## 7. Autoregressive decoder

Each decoder layer contains three major transformations:

1. causal self-attention over the generated prefix;
2. cross-attention (`encoder_attn`) to the multimodal encoder output;
3. a tokenwise FFN.

### 7.1 Decoder self-attention

Paths:

```text
language_model.model.decoder.layers.{i}.self_attn.q_proj
language_model.model.decoder.layers.{i}.self_attn.k_proj
language_model.model.decoder.layers.{i}.self_attn.v_proj
language_model.model.decoder.layers.{i}.self_attn.out_proj
```

**Verified role:** model dependencies among previously emitted target tokens under
a causal mask.

**Expected effect:** output grammar, `<poly>`/`</poly>` placement, alternating
X/Y location tokens, vertex ordering, avoiding repeated points, remembering the
polygon prefix, and deciding when to stop.

Self-attention can indirectly affect coordinates, but it cannot retrieve new
image evidence directly. It reasons over the generated sequence.

### 7.2 Decoder cross-attention (`encoder_attn`)

Paths:

```text
language_model.model.decoder.layers.{i}.encoder_attn.q_proj
language_model.model.decoder.layers.{i}.encoder_attn.k_proj
language_model.model.decoder.layers.{i}.encoder_attn.v_proj
language_model.model.decoder.layers.{i}.encoder_attn.out_proj
```

This is the most direct image-to-output routing point:

- decoder hidden states create queries;
- encoded image/prompt tokens create keys and values;
- each generated polygon token retrieves evidence from the image and prompt.

Expected module-level effects:

- `q_proj`: what the current output position asks the image/prompt for. A query
  after `<poly>` may learn to seek the next boundary vertex.
- `k_proj`: how encoder positions and prompt conditions are indexed for matching.
  This influences which room or boundary is selected.
- `v_proj`: what geometric/content information is returned from selected encoder
  states.
- `out_proj`: how retrieved evidence is remixed and written into the decoder
  residual stream before token prediction.

For referring-point polygon generation, this is the highest-priority language
attention subsystem. If parameter budget must be reduced, preserve
cross-attention capacity before decoder self-attention capacity.

### 7.3 Decoder FFN

Paths:

```text
language_model.model.decoder.layers.{i}.fc1  # 768 → 3072
language_model.model.decoder.layers.{i}.fc2  # 3072 → 768
```

**Verified role:** nonlinear tokenwise transformation after self- and
cross-attention.

**Expected effect:** convert retrieved image evidence and sequence context into
task-specific coordinate/token representations. Decoder FFNs are a plausible
next target when the model attends to the correct room but emits malformed or
poorly calibrated polygon sequences.

## 8. Embeddings and output head

### 8.1 Shared token embeddings

Florence-2 shares encoder input, decoder input, and output-token semantics. Prompt
words, `<poly>`, `</poly>`, and `<loc_0>` through `<loc_999>` already exist in the
checkpoint vocabulary; this repository does not add tokens.

**Expected effect if adapted:** alter the meaning/representation of prompt and
location tokens. Full embedding training would be expensive and risk broad
language drift. Selective trainable token rows could be considered only if
metrics show systematic location-token or delimiter confusion and the installed
PEFT version supports it safely with tied weights.

### 8.2 Learned language position embeddings

The encoder and decoder use learned absolute sequence positions.

**Expected effect if adapted:** sequence-length and output-position behavior.
These are not natural LoRA targets and should normally remain frozen.

### 8.3 LM head

Remote-code path:

```text
language_model.lm_head  # 768 → 51,289
```

The LM head maps decoder states to vocabulary logits and is tied to shared token
embeddings.

**Expected effect if adapted:** direct calibration of `<loc_*>`, polygon
delimiter, and EOS probabilities.

This can help token-choice problems, but it is broad and tied-weight handling
requires care. The current implementation correctly leaves it frozen. A LoRA on
the whole LM head would add roughly `r × (768 + 51,289)` trainable values and
affect all vocabulary logits, even though the task uses a narrow subset.

## 9. How LoRA changes a module

For a frozen linear weight `W`, ordinary PEFT LoRA computes:

```text
y = W x + (alpha / r) B A dropout(x)

A shape: [r, input_width]
B shape: [output_width, r]
```

The learned update `BA` has rank at most `r`. PEFT's default initialization sets
the B path to zero, so the adapter is initially a no-op and the model starts with
base-checkpoint behavior.

In the current configuration:

- vision rank: 8;
- encoder rank: 8;
- decoder rank: 16;
- alpha: `2 × rank` for every target;
- effective ordinary-LoRA scale: `alpha / rank = 2`;
- adapter dropout: 0.05;
- bias training: disabled.

Rank controls the maximum dimensionality of each learned update, not how strongly
it is applied. Alpha/rank controls update scaling. A larger rank is useful only
when the task needs more independent update directions and optimization/data can
use them.

## 10. Current repository implementation

### 10.1 Loaded checkpoint and task

Actual configuration:

```text
model_id: microsoft/Florence-2-base
image_size: 768
canvas_size: 1024
loc_bins: 1000
```

The source docstring says `Florence-2-base-ft`, but the configuration loads the
pretrained `Florence-2-base`. The distinction matters: the non-`-ft` checkpoint
has not received Microsoft's public downstream-task fine-tuning and may require
more adaptation for this instruction format.

The actual training sample is:

```text
Prompt:
Locate room polygon containing point <loc_x><loc_y>
[and optionally not containing other points]

Target:
<poly><loc_x0><loc_y0>...<loc_xn><loc_yn></poly>
```

Despite older README examples, the current target does not include a room name.
The task is polygon-only referring segmentation.

Current sampling settings produce 24 variations per selected floor-plan row:
2 room slots × 12 point variations. Five percent of variations request a point
outside every room and target an empty polygon. Excluded points are enabled with
10% probability.

### 10.2 Trainable target plan

Vision rank-8 LoRA:

```text
stage 2 blocks 5, 6, 7, 8
stage 3 block 0
```

For each selected block:

```text
spatial qkv
spatial proj
channel qkv
channel proj
```

Language encoder rank-8 LoRA:

```text
all 6 layers
all self-attention q_proj, k_proj, v_proj, out_proj
```

Language decoder rank-16 LoRA:

```text
all 6 layers
all self-attention q_proj, k_proj, v_proj, out_proj
all cross-attention q_proj, k_proj, v_proj, out_proj
```

Fully trainable:

```text
image_projection
image_proj_norm
```

Frozen:

```text
vision patch/downsampling and depthwise convolutions
vision stages 0 and 1
vision stage 2 blocks 0–4
all vision FFNs
all language FFNs
token and position embeddings
LM head
all base attention weights beneath LoRA
```

### 10.3 Target and parameter counts

LoRA-wrapped modules:

- vision: 5 blocks × 4 modules = 20;
- encoder: 6 layers × 4 modules = 24;
- decoder: 6 layers × 2 attention types × 4 modules = 48;
- total: 92 LoRA modules.

Approximate trainable values:

- vision LoRA: 294,912;
- encoder LoRA: 294,912;
- decoder LoRA: 1,179,648;
- total LoRA: 1,769,472;
- full image projection: 786,432;
- image projection LayerNorm: 1,536;
- total trainable: 2,557,440.

This is about 1.1% of the 231.4M base checkpoint. Approximately 69% of the
trainable budget is LoRA, 31% is the fully trained projector/norm, and about 46%
of the total budget is decoder LoRA.

### 10.4 Save/load behavior

The PEFT adapter saves LoRA weights and the trainable `image_proj_norm`.
`image_projection` is saved separately as:

```text
image_projection.pt
```

Inference must load both the PEFT adapter and this sidecar file. If the sidecar is
missing, loading silently leaves the base projector in place, producing a
plausible but incompletely adapted model.

### 10.5 Safety checks in the implementation

The implementation:

- freezes every base parameter first;
- constructs exact fully qualified target names;
- verifies every target exists before PEFT wrapping;
- checks the expected number of LoRA A matrices;
- rejects accidental trainable FFNs, embeddings, or LM head;
- explicitly checks early stage-2 vision blocks remain frozen;
- verifies the projector and saved norm are trainable.

These checks are a major strength. They prevent broad suffix matching from
quietly training unintended modules.

## 11. Graded review

### 11.1 Architecture and target correctness: 9.0/10

Strengths:

- Uses the correct sequence-to-sequence PEFT task type even though the checkpoint
  is loaded through `AutoModelForCausalLM`.
- Targets exact modules that exist in the Microsoft remote-code implementation.
- Covers the decoder's image-grounding cross-attention in all layers.
- Covers all multimodal encoder attention layers.
- Adapts both local spatial and global channel attention in selected vision
  blocks.
- Trains the otherwise inaccessible raw image-projection parameter.

Weakness:

- Layer counts and remote-code paths are hard-coded for Florence-2-base's current
  implementation. A large checkpoint or native Transformers conversion will not
  share all paths/counts.

### 11.2 Task fit: 8.8/10

Strengths:

- Region/pixel tasks need vision adaptation more than captioning. The Florence-2
  paper's own ablation found that freezing the vision encoder reduced COCO
  detection AP from 19.7 to 6.9 and referring-segmentation mIoU from 18.6 to 9.9.
- Late stage-2 plus stage-3 adaptation is a sensible balance between preserving
  generic edges and changing domain-level structure.
- Cross-attention receives the largest per-module rank, matching the need to
  ground generated coordinates in image evidence.
- The projector and norm directly address the floor-plan-to-language domain
  bridge.

Weaknesses:

- Thin walls, corners, and exact boundaries may benefit from earlier/more stage-2
  spatial adaptation or selected vision FFNs.
- All decoder attention receives rank 16. Decoder self-attention likely does not
  need the same capacity as decoder cross-attention for a narrow polygon grammar.

### 11.3 Parameter efficiency: 8.5/10

Strengths:

- Only about 1.1% of base-model size is trainable.
- More capacity is assigned to the decoder than encoder or vision tower.
- Early vision stages, embeddings, LM head, and all FFNs remain frozen.
- Constant alpha/rank keeps scaling comparable across rank-8 and rank-16 groups.

Weaknesses:

- The fully trained 786k projector is about 31% of trainable capacity and is not
  low rank.
- Decoder self-attention rank 16 may spend capacity on a simple output language
  that rank 8 could model.
- No evidence yet shows whether spreading rank-8 LoRA into selected FFNs would
  outperform rank 16 concentrated in decoder attention at a similar budget.

### 11.4 Training stability and regularization: 8.0/10

Strengths:

- LoRA dropout 0.05 is a conventional, modest regularizer.
- Alpha/rank = 2 is a reasonable standard-LoRA scale.
- Base weights are frozen before adapters are injected.
- bf16, gradient clipping at 1.0, warmup, and AdamW-style weight decay are
  reasonable training defaults.

Weaknesses:

- LoRA matrices and the fully trained projector use one global learning rate.
  The projector may prefer a lower rate than low-rank adapters.
- Standard LoRA is used rather than rank-stabilized LoRA. This is not a defect at
  ranks 8/16, but rsLoRA is a reasonable ablation if rank is increased.
- Ten epochs with no evaluation/early stopping can overfit repeated variations
  from the same floor plans.

### 11.5 Robustness and artifact integrity: 7.8/10

Strengths:

- Exact target-resolution and trainable-parameter validation fail fast.
- Projector save/load logic is explicit.
- Adapter loading uses the same base checkpoint and processor path.

Weaknesses:

- The projector sidecar is nonstandard and easy to omit when moving or publishing
  an adapter.
- Missing `image_projection.pt` is silently accepted during load.
- Remote-code module paths are not portable to native Transformers Florence-2.

### 11.6 Evidence and evaluation: 5.5/10

This is the largest gap.

- `eval_strategy="no"` means no held-out metric selects checkpoints.
- Training loss cannot distinguish valid-looking but geometrically wrong
  polygons from accurate polygons.
- There is no documented ablation comparing target sets or ranks.
- Beam-search output quality can differ from teacher-forced token loss.
- Repeated point variations from one floor plan are correlated; random
  sample-level splitting would leak floor-plan structure. Evaluation must split
  by `plan_id`.

### 11.7 Final grade: B+ (8.3/10)

The overall score weights the categories according to their importance for this
trainer:

```text
architecture and target correctness  30% × 9.0 = 2.70
task fit                            25% × 8.8 = 2.20
parameter efficiency                15% × 8.5 = 1.28
stability and regularization         10% × 8.0 = 0.80
robustness and artifact integrity    10% × 7.8 = 0.78
evidence and evaluation              10% × 5.5 = 0.55
                                               ────
                                               8.31
```

The adapter implementation is deliberate, selective, and strongly aligned with
the architecture. It earns a high implementation grade. It does not earn an A
because the setting remains a well-reasoned hypothesis rather than a
validation-proven optimum, and because artifact portability/integrity has
avoidable weaknesses.

## 12. Strengths of the current LoRA

1. **Cross-modal coverage is complete.** All decoder cross-attention projections
   in every layer are adapted.
2. **Vision is not incorrectly treated as optional.** Selected DaViT blocks adapt
   for a region/pixel-level task.
3. **Local and global vision mechanisms are both covered.** Spatial-window and
   channel-group attention receive LoRA.
4. **Capacity allocation is intentional.** Decoder rank 16 exceeds vision and
   encoder rank 8.
5. **The modality bridge is trainable.** Projector plus LayerNorm can align
   floor-plan features to the language hidden space.
6. **The update scale is consistent.** Alpha tracks rank, giving alpha/rank = 2
   everywhere.
7. **Target names are exact.** This avoids accidental matches among many repeated
   `proj`, `fc1`, and `qkv` names.
8. **Unexpected trainable weights are rejected.** The validation code makes the
   intended design enforceable.
9. **Output vocabulary already supports the task.** Existing Florence location
   and polygon tokens avoid random new-token initialization.

## 13. Weaknesses and risks

### 13.1 No validation feedback

The implementation cannot tell whether a different target selection is better,
whether ten epochs overfit, or which checkpoint gives the best geometry.

### 13.2 Decoder budget is not separated by attention type

Self-attention and cross-attention both receive rank 16. Their jobs differ:

- cross-attention retrieves image/prompt evidence;
- self-attention maintains polygon sequence state.

The former likely deserves more capacity for this task.

### 13.3 No FFN adaptation

Attention-only LoRA changes routing and content flow but leaves nonlinear feature
transformation fixed. If the base model can attend to floor-plan evidence but
cannot transform it into reliable room-boundary coordinates, attention-only
adaptation may plateau.

### 13.4 Vision selection favors later semantics

Stage-2 blocks 5–8 and stage 3 are reasonable for semantic structure, but early
stage-2 blocks remain fixed. Exact floor-plan boundaries may require adapting
more of the 48×48 stage while keeping stages 0/1 stable.

### 13.5 Fused vision `qkv` cannot express Q/V-only placement

Each vision adapter wraps `Linear(D, 3D)`. It can update all three slices. This is
not equivalent to the original LoRA paper's independently selected Q/V matrices.
It is not necessarily harmful, but should be understood when comparing target
counts or papers.

### 13.6 Full projector can dominate domain adaptation

The projector is high leverage and represents almost one-third of trainable
values. It can improve alignment quickly, but it may also absorb training signal
that could otherwise produce more spatially specific vision updates.

### 13.7 Checkpoint has a required sidecar

The PEFT directory alone is not the full trained model. Silent fallback when the
projector file is absent is a correctness risk.

### 13.8 Implementation and documentation drift

- Source docstring says `Florence-2-base-ft`; config uses `Florence-2-base`.
- Older README text says the vision encoder is frozen; selected vision blocks now
  have LoRA.
- Older README examples include a room-name target; current labels are polygon
  only.

These inconsistencies can cause future experiments to be interpreted incorrectly.

### 13.9 Remote-code portability

The current adapter targets Microsoft Hub remote-code paths. Native Transformers
uses different names and structure, including:

```text
remote vision_tower                         → native model.vision_tower
remote image_projection raw Parameter       → native multi-modal projector module
remote ...window_attn.fn.qkv                → native ...window_attn.qkv
remote language_model.model.encoder.layers  → native model.language_model.encoder.layers
remote language_model.lm_head               → native lm_head
```

Adapters and target configurations are not portable by simple path substitution;
weight layout and wrapping must also be verified.

## 14. Failure symptom to likely subsystem map

### Correct room, inaccurate or rounded boundary

Investigate:

1. stage-2 spatial window attention;
2. selected vision FFNs;
3. image resolution and final 24×24 token grid;
4. location-token quantization and label canonicalization;
5. decoder cross-attention values/output projection.

### Wrong room selected for a valid point

Investigate:

1. multimodal encoder Q/K routing;
2. decoder cross-attention Q/K;
3. late vision channel attention;
4. positive-point sampling distribution;
5. whether supplied loc-token points align with resized images.

### Model ignores excluded points

Investigate:

1. low frequency of excluded-point examples (`excluded_point_prob=0.1`);
2. encoder attention between prompt loc tokens and image tokens;
3. decoder cross-attention;
4. whether prompts contain enough hard cases with adjacent rooms.

This is more likely a data-frequency/fusion issue than an LM-head issue.

### Repeated vertices, invalid X/Y pairing, missing `</poly>`, or failure to stop

Investigate:

1. decoder self-attention;
2. decoder FFNs;
3. target canonicalization and truncation;
4. LM-head/location-token calibration;
5. beam-search constraints and maximum token length.

### Correct shape but globally shifted coordinates

Investigate:

1. image preprocessing and canvas/image scaling;
2. learned image positions;
3. projector alignment;
4. encoder/cross-attention positional use;
5. loc encode/decode formulas.

Do not assume more LoRA capacity will fix a deterministic preprocessing mismatch.

### Too many empty polygons or too few empty polygons

Investigate:

1. the 5% negative-sample prior;
2. negative-example difficulty;
3. decoder/LM-head token prior;
4. point containment consistency at polygon boundaries.

### Correct geometry but wrong room name

The current training labels do not contain room names. This behavior cannot be
learned from the current objective regardless of LoRA placement.

## 15. Recommended evaluation protocol

Before changing LoRA placement, create a held-out split by `plan_id`, not by
generated prompt variation. Every variation from one floor plan must stay in the
same split.

Track at least:

- polygon parse/validity rate;
- polygon IoU against the target room;
- boundary distance or vertex-to-edge distance;
- requested-point containment rate;
- excluded-point rejection rate;
- empty-polygon precision, recall, and F1;
- wrong-room rate;
- repeated-vertex rate;
- premature EOS and missing-close-tag rate;
- generation length;
- metrics by room size, vertex count, point-to-boundary distance, visual style,
  and number of excluded points.

Training token loss should remain a diagnostic, not the checkpoint-selection
criterion. Select checkpoints on a geometry-oriented validation score.

## 16. Recommended ablation sequence

Change one axis at a time and keep the total data split, seed, optimizer,
generation parameters, and checkpoint-selection metric fixed.

### Baseline A — current configuration

Keep:

- vision attention rank 8 on stage-2 blocks 5–8 and stage 3;
- encoder attention rank 8;
- decoder self/cross attention rank 16;
- full projector and norm.

This is the reference, not an assumed optimum.

### Ablation B — split decoder capacity

Try:

- decoder cross-attention rank 16;
- decoder self-attention rank 8;
- everything else unchanged.

Goal: determine whether rank-16 self-attention is unnecessary. Expected outcome:
similar geometry and grammar with fewer parameters, unless long/complex polygon
prefixes truly need the extra sequence capacity.

### Ablation C — decoder FFN

Add decoder `fc1`/`fc2` LoRA at rank 4 or 8 while keeping cross-attention rank 16.
Compare at a controlled parameter budget, potentially reducing decoder
self-attention rank.

Goal: improve token transformations, output validity, and coordinate calibration.

### Ablation D — selected vision FFNs

Add spatial and channel FFN `fc1`/`fc2` only in the already selected late vision
blocks, initially at rank 4 or 8.

Goal: test whether feature-remapping capacity improves boundary accuracy after
attention has selected the correct structure.

### Ablation E — broaden stage-2 spatial adaptation

Extend spatial `qkv`/`proj` to stage-2 blocks 0–4 before touching stages 0/1.
Optionally leave additional channel attention frozen for a focused test.

Goal: improve 48×48-grid boundary structure while preserving low-level feature
extraction.

### Ablation F — projector learning rate

Use a lower learning rate for the full projector than for LoRA weights.

Goal: retain high-leverage modality alignment while reducing projector drift.

### Ablation G — base versus base-ft

Compare `microsoft/Florence-2-base` against
`microsoft/Florence-2-base-ft` using the same target plan.

Goal: determine whether downstream-tuned polygon/grounding behavior offers a
better initialization than the pure pretrained checkpoint.

Do not assume `base-ft` is automatically superior: the task prompt and floor-plan
domain differ from Microsoft's public downstream mixture, so this requires
measurement.

## 17. Where not to put LoRA first

Avoid these as first-line changes:

- temporal embeddings for a single-image task;
- all language position embeddings;
- all shared token embeddings;
- a broad full-vocabulary LM-head update;
- stage-0/1 vision modules without evidence of low-level domain failure;
- every linear layer through a generic suffix list.

Broadening targets can improve capacity, but it also removes the ability to
identify which subsystem fixed the problem.

## 18. Important `all-linear` PEFT edge case

PEFT documents `target_modules="all-linear"` as selecting linear/Transformers
`Conv1D` modules and normally excluding a `PreTrainedModel` output head.

For the exact Microsoft remote Florence-2 model inspected with the repository
environment:

- vision `qkv`, `proj`, and FFNs are selected;
- language attention and FFNs are selected;
- `language_model.lm_head` is also selected because the remote top-level model
  does not report it as its output embedding in the way PEFT expects;
- `image_projection` is missed because it is a raw parameter;
- patch and depthwise `Conv2d` modules are missed.

Therefore `all-linear` is neither "all desired adaptation points" nor safely
equivalent to "all transformer linears except the head" here. The current explicit
fully qualified target list is safer.

## 19. Source hierarchy and version caveat

This repository pins the older remote-code-compatible Transformers range and
loads with `trust_remote_code=True`. Florence-2 later gained native Transformers
support with a changed hierarchy.

Always determine which implementation produced an adapter:

1. Microsoft Hub remote code;
2. native Transformers Florence-2;
3. a third-party converted checkpoint.

Do not copy LoRA module names among these implementations without enumerating
`named_modules()` and verifying weight shapes.

## 20. Sources

Primary and upstream:

- Florence-2 CVPR paper:
  https://openaccess.thecvf.com/content/CVPR2024/papers/Xiao_Florence-2_Advancing_a_Unified_Representation_for_a_Variety_of_Vision_Tasks_CVPR_2024_paper.pdf
- Florence-2 arXiv HTML:
  https://arxiv.org/html/2311.06242
- Microsoft Florence-2-base model card:
  https://huggingface.co/microsoft/Florence-2-base
- Microsoft Florence-2-base configuration:
  https://huggingface.co/microsoft/Florence-2-base/blob/main/config.json
- Microsoft remote-code implementation:
  https://huggingface.co/microsoft/Florence-2-base/blob/main/modeling_florence2.py
- Native Transformers Florence-2 implementation:
  https://github.com/huggingface/transformers/blob/main/src/transformers/models/florence2/modeling_florence2.py
- Transformers native-integration PR:
  https://github.com/huggingface/transformers/pull/38188
- Transformers Florence-2 documentation:
  https://huggingface.co/docs/transformers/model_doc/florence2
- DaViT paper:
  https://arxiv.org/abs/2204.03645
- Official DaViT repository:
  https://github.com/dingmyu/davit
- Original LoRA paper:
  https://arxiv.org/abs/2106.09685
- Microsoft LoRA implementation:
  https://github.com/microsoft/LoRA
- PEFT LoRA reference:
  https://huggingface.co/docs/peft/en/package_reference/lora
- PEFT LoRA developer guide:
  https://huggingface.co/docs/peft/main/developer_guides/lora

Relevant repository files:

- `backend/florence_trainer/model.py`
- `backend/florence_trainer/config.yaml`
- `backend/florence_trainer/config.py`
- `backend/florence_trainer/data.py`
- `backend/florence_trainer/train.py`
- `backend/florence_trainer/loc.py`
- `backend/florence_trainer/inference/generate.py`
- `backend/florence_trainer/README.md`

Community reports about Florence-2 LoRA target choices are useful hypotheses, not
architecture guarantees. They should not outweigh this project's own
plan-level, geometry-based ablations.
