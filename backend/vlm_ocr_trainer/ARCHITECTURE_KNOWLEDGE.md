# OCR Architecture Knowledge

Research-backed comparison of `OcrVlm` with compact and representative OCR
architectures. Parameter counts and accuracy figures come from the linked
papers or official model documentation.

## Comparison caveat

Published accuracy numbers are not directly comparable with this project.
Most referenced models recognize cropped words or single text lines, while
`OcrVlm` transcribes up to six lines from a two-dimensional canvas. Datasets,
vocabularies, image sizes, training data, and metrics also differ.

## Current OcrVlm

**Release:** local model, 2026  
**Parameters:** 929,800  
**Task:** grayscale, multiline image-to-character transcription

```text
128–256 px grayscale image
  → depthwise-separable CNN stem, total stride 16
  → 4 vision Transformer blocks
       width 96, 4 heads, MLP 192, axial 2D RoPE
  → projection 96 → 128
  → 3 autoregressive decoder blocks
       width 128, 4 heads, MLP 256, 1D RoPE
  → tied character embedding/language head
  → 99-token character vocabulary, including newline
```

Parameter allocation:

- CNN stem: 13,512 (1.5%)
- Vision blocks: 297,600 (32.0%)
- Decoder blocks: 593,280 (63.8%)
- Remaining projection, normalization, and embedding: 25,408 (2.7%)

Strengths:

- Preserves a two-dimensional representation for multiline reading order.
- Uses local CNN features followed by global visual attention.
- Parameter-free 2D and 1D rotary position embeddings.
- Autoregressive decoding models character and newline dependencies.
- Weight tying keeps the character embedding and output head inexpensive.

Main limitations:

- Four stride-2 operations reduce glyphs to stride-16 features before visual
attention.
- The 13.5K-parameter stem has little capacity for handwriting and fine
strokes.
- Nearly two-thirds of parameters are spent in the decoder.
- All vision blocks use global attention; there is no explicit local
character-component mixer after the stem.
- Greedy decoding can propagate errors or hallucinate plausible characters.
- Mixed-size padded batches require correct visual masking or size bucketing.

## PP-OCRv6 Tiny

**Release:** June 2026  
**Parameters:** 1.1M recognizer; 1.5M with detector  
**Reported recognition accuracy:** 73.5% on PaddleOCR's weighted multilingual,
multi-scenario benchmark  
**Source:** [PP-OCRv6](https://arxiv.org/abs/2606.13108)

```text
3×48×W text-line image
  → PPLCNetV4/LCNetV4 convolutional backbone
  → asymmetric height-only downsampling
  → [B, 160, 1, W] feature sequence
  → direct reshape and projection
  → CTC head for inference

Training only:
  → auxiliary NRTR attention decoder
  → knowledge distillation from a larger teacher
```

### Compared with OcrVlm

- PP-OCR preserves horizontal resolution; OcrVlm reduces width by 16.
- PP-OCR dedicates deployed capacity to visual extraction; OcrVlm dedicates
63.8% to its language decoder.
- PP-OCR decodes all CTC positions in parallel; OcrVlm decodes sequentially.
- PP-OCR uses language supervision and distillation during training without
retaining that cost at inference.
- PP-OCR assumes a cropped text line. OcrVlm can learn multiple lines and
newline order from a full 2D canvas.

**Transferable ideas:** stronger visual backbone, dimension-aware
downsampling, training-only decoder supervision, and teacher distillation.
Its one-dimensional CTC head is not a direct replacement for multiline output.

## HTR-JAND Student

**Release:** December 2024  
**Parameters:** 0.75M student; 1.5M teacher  
**Reported student result:** approximately 4.12% CER on IAM  
**Source:** [HTR-JAND](https://arxiv.org/abs/2412.18524)

```text
68×864 grayscale handwriting line
  → 3 gated CNN and squeeze-and-excitation blocks
  → 4 bidirectional LSTM layers, hidden size 64
  → self-attention and Proxima attention
  → CTC
```

### Compared with OcrVlm

- HTR-JAND uses gated local extraction and channel recalibration; OcrVlm uses
basic depthwise-separable stem blocks.
- HTR-JAND models a horizontal sequence with BiLSTMs and CTC; OcrVlm uses 2D
vision attention and an autoregressive decoder.
- HTR-JAND is specialized for long handwriting lines and distilled from a
teacher.
- OcrVlm is more flexible for multiline layout but has weaker local stroke
processing.

**Transferable ideas:** gated local features and distillation. The recurrent,
one-dimensional backend is less suitable for full-canvas transcription.

## SVTR-Tiny

**Release:** May 2022; IJCAI 2022  
**Parameters:** 4.15M backbone; approximately 6.03M evaluated recognizer  
**Reported accuracy:** 84.1–96.3% across six English scene-text benchmarks,
approximately 89.3% unweighted average  
**Source:** [SVTR](https://arxiv.org/abs/2205.00159)

```text
text-line image
  → overlapping patch embedding at stride 4
  → stage 1: width 64, 3 blocks
  → stage 2: width 128, 6 blocks
  → stage 3: width 256, 3 blocks
  → six local mixers followed by six global mixers
  → height collapsed to one
  → 192-dimensional sequence
  → linear CTC prediction
```

### Compared with OcrVlm

- SVTR begins character-component processing at stride 4; OcrVlm begins
attention at stride 16.
- SVTR is hierarchical and widens with depth; OcrVlm remains at width 96.
- SVTR uses explicit local-before-global mixing; OcrVlm uses four uniform
global Transformer blocks.
- SVTR spends its capacity on 12 visual blocks and uses no autoregressive
decoder.
- SVTR is designed for text lines; OcrVlm retains the vertical dimension for
multiple lines.

**Transferable idea:** preserve fine features and process local strokes before
global reading context.

## PARSeq-Ti and PARSeq-S

**Release:** July 2022; ECCV 2022  
**Parameters:** approximately 5M for Ti; 23.8M for S  
**Reported PARSeq-S accuracy:** 91.9% with synthetic training and 96.0% with
real training; these figures do not describe PARSeq-Ti  
**Source:** [PARSeq](https://arxiv.org/abs/2207.06966)

PARSeq-S configuration:

```text
32×128 word image
  → 4×8 patch embedding
  → 12-layer ViT encoder
       width 384, 6 heads
  → one-layer visio-lingual decoder
       learned output-position queries, 12 heads
  → autoregressive, non-autoregressive, or iterative decoding
```

PARSeq trains shared decoder weights with permutation language modelling.

### Compared with OcrVlm

- PARSeq uses a deep encoder and one shallow decoder; OcrVlm uses four encoder
and three decoder layers.
- Approximately 90% of PARSeq-S parameters belong to the visual encoder,
versus about 34% of OcrVlm before the decoder.
- PARSeq uses fine anisotropic patches; OcrVlm uniformly downsamples by 16.
- PARSeq uses learned output-position queries and permutation masks; OcrVlm
embeds the generated prefix and always trains left-to-right.
- PARSeq can share one model across several decoding modes. OcrVlm only uses
greedy autoregressive decoding.
- Both retain image-to-text cross-attention, but OcrVlm additionally learns
multiline newline order.

**Transferable idea:** a deep visual encoder with a single decoder layer is the
strongest precedent for rebalancing OcrVlm without giving up autoregressive
multiline transcription.

## ViTSTR-Tiny

**Release:** May 2021; ICDAR 2021  
**Parameters:** 5.4M  
**Reported accuracy:** 80.3% aggregate word accuracy; 82.1% with augmentation  
**Source:** [ViTSTR](https://arxiv.org/abs/2105.08582)

```text
224×224 grayscale image
  → non-overlapping 16×16 patches
  → 12-layer DeiT-Tiny encoder
       width 192, 3 heads
  → fixed position-wise character prediction
```

### Compared with OcrVlm

- Both reach 16-pixel visual granularity.
- ViTSTR uses direct patch projection; OcrVlm uses a CNN stem.
- ViTSTR has 12 wider visual layers and no cross-attention decoder.
- ViTSTR predicts bounded output positions in parallel; OcrVlm supports
variable-length autoregressive text.
- ViTSTR benefits from DeiT initialization; OcrVlm trains from scratch on
domain-specific synthetic data.

ViTSTR supports the visual-first allocation principle, but fixed parallel
positions are less natural for variable-length multiline transcription.

## CRNN

**Release:** July 2015 preprint; 2017 journal publication  
**Parameters:** approximately 8.3M  
**Reported unified reproduction:** approximately 78.4% aggregate accuracy,
with individual benchmarks around 64.2–93.1%  
**Sources:** [CRNN](https://arxiv.org/abs/1507.05717) and
[unified comparison](https://arxiv.org/abs/1904.01906)

```text
text-line image
  → VGG-style CNN
  → height collapse into column features
  → 2 bidirectional LSTM layers
  → per-column character probabilities
  → CTC
```

### Compared with OcrVlm

- CRNN reduces OCR to a left-to-right sequence; OcrVlm keeps a 2D grid.
- CRNN uses recurrent sequence modelling and monotonic CTC alignment.
- OcrVlm uses global visual attention and arbitrary decoder cross-attention.
- CRNN handles long cropped lines efficiently and avoids AR exposure bias.
- OcrVlm can represent several lines and language-conditioned reading order.

CRNN remains a useful baseline if floor-plan labels are detected and cropped
before recognition. It is not a direct full-canvas replacement.

## ABINet

**Release:** March 2021; CVPR 2021  
**Parameters:** approximately 32.8–36.7M  
**Reported accuracy:** approximately 89.8–92.6%, depending on data and
iterative refinement  
**Source:** [ABINet](https://arxiv.org/abs/2103.06495)

```text
image
  → ResNet backbone
  → visual Transformer
  → position-attention character queries
  → parallel visual character probabilities
  → 4-layer bidirectional cloze language model
       width 512, 8 heads, MLP 2048
  → visual-language fusion
  → optional iterative correction
```

### Compared with OcrVlm

- ABINet explicitly separates visual recognition and language correction.
- OcrVlm performs both through one autoregressive cross-attention path.
- ABINet's language model sees left and right context; OcrVlm only sees the
generated prefix.
- ABINet can block gradients between visual and language modules, reducing the
chance that language priors override image evidence.
- ABINet predicts in parallel and optionally refines; OcrVlm generates once
from left to right.
- ABINet's dimensions are far beyond the available parameter budget.

A training-only bidirectional consistency teacher is more realistic than
embedding an ABINet-style language model in OcrVlm.

## TrOCR-Small

**Release:** September 2021 preprint; AAAI 2023  
**Parameters:** approximately 62M  
**Reported results:** 4.22% CER on IAM and 95.86 F1 on SROIE  
**Source:** [TrOCR](https://arxiv.org/abs/2109.10282)

```text
384×384 image
  → 16×16 patch embedding
  → DeiT-Small encoder
       12 layers, width 384, 6 heads
  → MiniLM-derived autoregressive decoder
       6 layers, width 256, 8 heads
  → wordpiece prediction with beam search
```

### Compared with OcrVlm

- TrOCR is the closest macro-architecture: image Transformer followed by an
autoregressive text Transformer.
- TrOCR uses direct patches; OcrVlm adds a lightweight local CNN stem.
- TrOCR has 12 encoder and 6 decoder layers, compared with four and three.
- TrOCR uses large-scale visual and language pretraining; OcrVlm trains from
scratch.
- TrOCR emits wordpieces and normally uses beam search; OcrVlm emits
characters with greedy decoding.
- OcrVlm uses parameter-free 2D/1D RoPE and is roughly 67 times smaller.

OcrVlm is effectively an aggressively compressed TrOCR, but its compression
leaves a disproportionate amount of capacity in the decoder.

## Architectural conclusions

### Most relevant references

1. **PARSeq:** preserve the encoder-decoder formulation but use a deep encoder
  and one shallow decoder.
2. **PP-OCRv6 Tiny:** spend the near-1M budget on visual extraction and use
  training-only language guidance or distillation.
3. **SVTR:** process fine local character components before global context.
4. **HTR-JAND:** use stronger gated local processing for handwriting.
5. **TrOCR:** confirms the encoder-decoder formulation, but also shows how much
  it normally depends on scale and pretraining.

### Recommended fixed-budget synthesis

```text
grayscale multiline image
  → stronger stride-4 and stride-8 local CNN stages
  → cheap depthwise local character-component mixing
  → downsample or fuse to stride 16
  → 6–8 global 2D vision blocks
  → one shallow autoregressive cross-attention decoder
  → character and newline output
```

This is a reallocation of the existing parameter budget, not a proposal to
make the model materially larger. The central idea is to move capacity from
text generation into visual recognition. At present, three decoder blocks use
593,280 parameters, while the complete CNN stem uses only 13,512. Removing two
decoder blocks saves 395,520 parameters. Increasing the vision encoder from
four to eight existing-width blocks costs 297,600 parameters, leaving roughly
97,900 parameters for a stronger stem and local visual mixing while remaining
within the current 929,800-parameter budget.

The exact number of vision blocks should be selected by validation rather than
assumed in advance. Six blocks leave more capacity for the CNN stages; eight
blocks maximize global visual reasoning. Both are plausible fixed-budget
designs.

#### Stronger stride-4 and stride-8 local CNN stages

`Stride` describes how much the image resolution has been reduced. A stride-4
feature map has one feature position for each 4×4 region of the input; a
stride-8 map has one for each 8×8 region. These relatively high-resolution
maps still contain thin strokes, corners, gaps, and nearby character parts.

The current stem downsamples four times in quick succession and reaches
stride 16 before substantial processing occurs. A stronger stem would:

- spend more convolutional layers at stride 4 and stride 8;
- increase channels gradually instead of discarding spatial detail early;
- use residual connections so added local processing is easy to optimize; and
- optionally use gates or squeeze-and-excitation to suppress background and
emphasize useful stroke channels.

This should help most with small fonts, handwriting, low contrast, and similar
characters such as `O/0`, `I/l/1`, and `c/e`. The trade-off is higher compute
and activation memory because convolution happens on larger feature maps.

#### Cheap depthwise local character-component mixing

A normal convolution mixes spatial positions and channels simultaneously and
can be expensive. A depthwise-separable block first applies a small spatial
filter independently to each channel and then uses a 1×1 convolution to mix
channels. It therefore gives nearby pixels and stroke fragments more chances
to interact at low parameter cost.

“Character-component mixing” does not require character bounding boxes. It
means that the network combines local evidence such as vertical strokes,
curves, junctions, and dots before asking global attention to reason about the
whole page. Two or more residual depthwise blocks at stride 8 are a reasonable
starting point. This follows SVTR's local-before-global principle while
retaining OcrVlm's two-dimensional feature grid.

#### Downsample or fuse to stride 16

Global self-attention has quadratic cost in the number of visual tokens.
Keeping the entire encoder at stride 8 would create four times as many tokens
as stride 16 and roughly sixteen times the attention-score work. The model
should therefore perform rich local processing at stride 8, then reduce the
map to stride 16 before global attention.

Two variants are possible:

1. **Downsample:** transform the stride-8 map directly into a stride-16 map.
  This is simplest and cheapest.
2. **Fuse:** combine the new stride-16 map with a projected summary of the
  stride-8 features. This may preserve more fine detail, but adds memory,
   implementation complexity, and parameters.

Direct downsampling is the recommended first implementation. Fusion should
only be added if evaluation shows that fine strokes are still being lost.

#### Six to eight global 2D vision blocks

After local features are formed, 2D self-attention lets every image region
use evidence from every other region. This is useful for assigning characters
to lines, distinguishing text from background, and establishing multiline
reading order. The existing axial 2D RoPE already tells attention about row
and column displacement, so it can be retained.

Making the encoder deeper gives visual tokens several rounds of global
reasoning. This is a better use of the fixed budget when recognition errors
originate in the image. It does not necessarily require a wider encoder:
keeping width 96 and adding blocks is parameter-efficient and preserves the
existing projection interface.

The cost is slower image encoding. However, image encoding runs once, whereas
an autoregressive decoder runs repeatedly as the output grows. Moving
parameters from decoder depth to encoder depth can therefore improve both
visual capacity and long-sequence inference efficiency.

#### One shallow autoregressive cross-attention decoder

An autoregressive decoder predicts the next character from earlier predicted
characters and the encoded image. `Cross-attention` is the operation that lets
each output position select relevant visual regions. Retaining one decoder
block preserves:

- dependence on the image at every output step;
- character-level language context;
- explicit newline generation; and
- flexible output length terminated by `<EOS>`.

Reducing three blocks to one limits the decoder's ability to act as a powerful  
standalone language model. That is intentional: an oversized decoder can  
produce linguistically plausible text even when the visual evidence is weak,  
causing substitutions or hallucinations. The risk is that one block may be  
insufficient for difficult reading-order cases. Validation should therefore  
compare one and two blocks; a second block should only be retained if it  
improves full-canvas CER enough to justify taking capacity from the encoder.

#### Practical first candidate

A conservative first candidate is:

```text
CNN: retain stride 16, but add most new local capacity at stride 4 and 8
Local mixer: 2 residual depthwise-separable blocks at stride 8
Vision: width 96, 4 heads, MLP 192, 8 global blocks
Decoder: width 128, 4 heads, MLP 256, 1 block
Positions/output: existing 2D RoPE, 1D RoPE, vocabulary, and tied output head
```

Relative to the current model, the decoder change frees 395,520 parameters and
the four added vision blocks consume 297,600. The strengthened CNN and local
mixer must therefore add no more than about 97,900 parameters if the strict
current parameter ceiling is retained.

This recommendation is a research-backed hypothesis, not a guaranteed
improvement. It should be evaluated by changing one factor at a time and
reporting at least character error rate, exact sequence accuracy, newline
accuracy, latency, and peak memory. Comparisons must use the same training
steps and data because moving capacity cannot compensate for incorrect visual
padding masks, label truncation, or inconsistent synthetic labels.

This combines:

- PP-OCRv6's visual-first parameter allocation.
- SVTR's local-before-global processing.
- PARSeq's deep-encoder/shallow-decoder structure.
- OcrVlm's 2D representation and multiline autoregressive output.

CTC is preferable when every input is a cropped text line. For the current
full-canvas task, retaining one shallow autoregressive decoder is safer because
line discovery and newline order are part of the prediction problem.

## Research references

- [CRNN: An End-to-End Trainable Neural Network for Image-Based Sequence Recognition](https://arxiv.org/abs/1507.05717)
- [What Is Wrong With Scene Text Recognition Model Comparisons?](https://arxiv.org/abs/1904.01906)
- [ABINet: Read Like Humans](https://arxiv.org/abs/2103.06495)
- [ViTSTR: Vision Transformer for Fast and Efficient Scene Text Recognition](https://arxiv.org/abs/2105.08582)
- [TrOCR: Transformer-Based Optical Character Recognition](https://arxiv.org/abs/2109.10282)
- [SVTR: Scene Text Recognition with a Single Visual Model](https://arxiv.org/abs/2205.00159)
- [PARSeq: Scene Text Recognition with Permuted Autoregressive Sequence Models](https://arxiv.org/abs/2207.06966)
- [HTR-JAND](https://arxiv.org/abs/2412.18524)
- [SVTRv2: CTC Beats Encoder-Decoder Models](https://arxiv.org/abs/2411.15858)
- [PP-OCRv6](https://arxiv.org/abs/2606.13108)

