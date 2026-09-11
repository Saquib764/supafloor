"""Print OCR VLM architecture, parameter counts, and dummy-forward shapes."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import torch

from canvas import input_output_row, text_panel
from config import CONFIG_PATH, Config, load_config
from data import image_to_tensor, render_text_sample
from device import training_device
from model import (
    DECODER_DIM,
    DECODER_HEADS,
    DECODER_LAYERS,
    DECODER_MLP,
    OcrVlm,
    VISION_DIM,
    VISION_HEADS,
    VISION_LAYERS,
    VISION_MLP,
)
from vocab import CharVocab

_PREVIEW_DIR = CONFIG_PATH.parent / "_preview"


def _fmt_params(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    if n >= 1_000:
        return f"{n / 1e3:.2f}K"
    return str(n)


def _fmt_mb(num_bytes: int) -> str:
    return f"{num_bytes / (1024 ** 2):.2f} MB"


def _fmt_line(n: int, num_bytes: int) -> str:
    return f"{n:,} ({_fmt_params(n)}, {_fmt_mb(num_bytes)})"


_LAYER_ORDER = (
    "stem",
    "vision_blocks.0",
    "vision_blocks.1",
    "vision_blocks.2",
    "vision_blocks.3",
    "vision_norm",
    "vision_proj",
    "token_embedding",
    "decoder_blocks.0",
    "decoder_blocks.1",
    "decoder_blocks.2",
    "decoder_norm",
)


def _print_architecture_summary(cfg: Config, vocab: CharVocab) -> None:
    width, height = cfg.data.image_sizes[0]
    s1w, s1h = width // 2, height // 2
    s2w, s2h = width // 4, height // 4
    s3w, s3h = width // 8, height // 8
    s4w, s4h = width // 16, height // 16

    print("\n=== Architecture ===", flush=True)
    print("Task:     grayscale text image → character transcription (autoregressive)", flush=True)
    sizes = ", ".join(f"{w}×{h}" for w, h in cfg.data.image_sizes)
    print(f"Input:    (1, H, W)   gray, normalized [-1, 1]   sizes: {sizes}", flush=True)
    print(f"Output:   (seq, {vocab.size})   char logits → BOS…text…EOS", flush=True)
    print(flush=True)
    print("OcrVlm", flush=True)
    print("├─ CnnStem", flush=True)
    print(f"│  ├─ conv 3×3 stride 2     1→24 ch   @ {s1w}×{s1h}", flush=True)
    print(f"│  ├─ dwsep stride 2        24→48 ch  @ {s2w}×{s2h}", flush=True)
    print(f"│  ├─ dwsep stride 2        48→72 ch  @ {s3w}×{s3h}", flush=True)
    print(f"│  └─ dwsep stride 2        72→96 ch  @ {s4w}×{s4h}", flush=True)
    print(
        f"├─ VisionEncoder ({VISION_LAYERS} layers, d={VISION_DIM}, "
        f"heads={VISION_HEADS}, mlp={VISION_MLP}, 2D RoPE)",
        flush=True,
    )
    for index in range(VISION_LAYERS):
        branch = "│  └─" if index == VISION_LAYERS - 1 else "│  ├─"
        print(f"{branch} block {index}   bidirectional self-attn (variable tokens)", flush=True)
    print(f"├─ vision_proj            {VISION_DIM}→{DECODER_DIM}", flush=True)
    print(
        f"├─ TextDecoder ({DECODER_LAYERS} layers, d={DECODER_DIM}, "
        f"heads={DECODER_HEADS}, mlp={DECODER_MLP}, 1D RoPE)",
        flush=True,
    )
    print(f"│  ├─ token_embedding     {vocab.size}×{DECODER_DIM} (weight-tied lm_head)", flush=True)
    for index in range(DECODER_LAYERS):
        branch = "│  └─" if index == DECODER_LAYERS - 1 else "│  ├─"
        print(
            f"{branch} block {index}   causal self-attn + cross-attn → visual tokens",
            flush=True,
        )
    print(f"└─ lm_head                {DECODER_DIM}→{vocab.size} (tied)", flush=True)


def _print_input_summary(cfg: Config, vocab: CharVocab) -> None:
    sizes = ", ".join(f"{w}×{h}" for w, h in cfg.data.image_sizes)
    example_w, example_h = cfg.data.image_sizes[0]
    grid_w, grid_h = example_w // 16, example_h // 16
    visual_tokens = grid_w * grid_h

    print("\n=== Input ===", flush=True)
    print("Source:   synthetic grayscale PIL image (random OCR-style text lines)", flush=True)
    print(f"Canvas:   one of [{sizes}] px (width×height), chosen per sample", flush=True)
    print(
        f"Text:     {list(cfg.data.line_counts)} lines (sampled), "
        f"{cfg.data.sentence_len_min}–{cfg.data.sentence_len_max} chars total",
        flush=True,
    )
    print(flush=True)
    print("Preprocess pipeline:", flush=True)
    print("  1. PIL grayscale (L) → float32 numpy", flush=True)
    print("  2. Normalize:  x = pixel / 127.5 − 1   (range [-1, 1])", flush=True)
    print("  3. Tensor shape:  [1, H, W]", flush=True)
    print("  4. Batch: pad to max H×W in batch with 1.0 (white)", flush=True)
    print(flush=True)
    print("Vision encode (fixed for the whole decode loop):", flush=True)
    print(f"  image [1, 1, H, W]", flush=True)
    print(f"    → CNN stem ÷16", flush=True)
    print(f"    → ViT encoder + 2D RoPE", flush=True)
    print(f"    → linear {VISION_DIM}→{DECODER_DIM}", flush=True)
    print(
        f"    → visual tokens [1, H/16·W/16, {DECODER_DIM}]"
        f"   (e.g. {example_w}×{example_h} → {grid_w}×{grid_h} = {visual_tokens} tokens)",
        flush=True,
    )
    print(flush=True)
    print("Training label (teacher forcing):", flush=True)
    print('  text  "Room 203\\nWidth: 4.5m"', flush=True)
    print(f"  ids   [<BOS>, R, o, o, m, …, <EOS>]   (max {cfg.max_seq_len} tokens)", flush=True)
    print(f"  loss  cross-entropy on next-char prediction; <PAD> ignored", flush=True)


def _print_prediction_summary(cfg: Config, vocab: CharVocab) -> None:
    print("\n=== Prediction ===", flush=True)
    print("Inference: greedy autoregressive decode (model.generate)", flush=True)
    print(flush=True)
    print("  1. Encode image once → visual tokens (cached for all steps)", flush=True)
    print(f"  2. Start sequence with <BOS>  (id {vocab.bos_id})", flush=True)
    print("  3. Repeat until <EOS> or max length:", flush=True)
    print("       logits = decoder(visual, generated_tokens)", flush=True)
    print("       next   = argmax(logits[:, -1])", flush=True)
    print("       append next to generated_tokens", flush=True)
    print(f"  4. Stop at <EOS>  (id {vocab.eos_id})  or max_seq_len={cfg.max_seq_len}", flush=True)
    print("  5. Decode token ids → string (skip <BOS>, <EOS>, <PAD>)", flush=True)
    print(flush=True)
    print("Example generation trace:", flush=True)
    print("  step 0:  <BOS>                          → predict 'R'", flush=True)
    print("  step 1:  <BOS> R                        → predict 'o'", flush=True)
    print("  step 2:  <BOS> R o                      → predict 'o'", flush=True)
    print("  …", flush=True)
    print("  step N:  <BOS> Room 203\\nWidth: 4.5m   → predict <EOS>  (done)", flush=True)
    print(flush=True)
    print("Decoder at each step:", flush=True)
    print("  • causal self-attention over text tokens (1D RoPE)", flush=True)
    print("  • cross-attention to all visual tokens", flush=True)
    print("  • lm_head → softmax over 99 char tokens", flush=True)


def _match_layer(name: str) -> str:
    if name.startswith("lm_head."):
        return "token_embedding"
    matched = next(
        (key for key in _LAYER_ORDER if name == key or name.startswith(f"{key}.")),
        None,
    )
    if matched is None:
        raise RuntimeError(f"parameter not in known layers: {name}")
    return matched


def _layer_param_counts(model: OcrVlm) -> dict[str, tuple[int, int]]:
    counts = {name: (0, 0) for name in _LAYER_ORDER}
    seen: set[int] = set()
    for name, param in model.named_parameters():
        layer = _match_layer(name)
        storage = param.untyped_storage().data_ptr()
        if storage in seen:
            continue
        seen.add(storage)
        count, size = counts[layer]
        n = param.numel()
        counts[layer] = (count + n, size + n * param.element_size())
    return counts


def print_model(
    cfg: Config | None = None,
    vocab: CharVocab | None = None,
    *,
    dummy_forward: bool = True,
) -> OcrVlm:
    cfg = cfg or load_config()
    vocab = vocab or CharVocab()
    model = OcrVlm(vocab)
    model.eval()

    layers = _layer_param_counts(model)
    total = trainable = 0
    total_bytes = trainable_bytes = 0
    for param in model.parameters():
        n = param.numel()
        nbytes = n * param.element_size()
        total += n
        total_bytes += nbytes
        if param.requires_grad:
            trainable += n
            trainable_bytes += nbytes

    _print_architecture_summary(cfg, vocab)
    _print_input_summary(cfg, vocab)
    _print_prediction_summary(cfg, vocab)
    print("\n=== Parameters ===", flush=True)
    for name in _LAYER_ORDER:
        count, size = layers[name]
        print(f"{name + ':':<22} {_fmt_line(count, size)}", flush=True)
    print(f"{'trainable:':<22} {_fmt_line(trainable, trainable_bytes)}", flush=True)
    print(f"{'total:':<22} {_fmt_line(total, total_bytes)}", flush=True)

    if dummy_forward:
        fw_w, fw_h = cfg.data.image_sizes[0]
        image = torch.zeros(1, 1, fw_h, fw_w)
        token_ids = torch.tensor([[vocab.bos_id, vocab.stoi["H"], vocab.stoi["i"]]], dtype=torch.long)
        full_ids = torch.cat([token_ids, torch.tensor([[vocab.eos_id]])], dim=1)
        with torch.no_grad():
            features = model.stem(image)
            visual = model.encode_image(image)
            logits = model(image, full_ids)
        generated = model.generate(image, max_length=8)
        print("\n=== Dummy forward ===")
        print(f"input tensor:     {tuple(image.shape)}  float32, normalized")
        print(f"stem features:    {tuple(features.shape)}")
        print(f"visual tokens:    {tuple(visual.shape)}")
        print(f"teacher-forced:   tokens {full_ids[0].tolist()}")
        print(f"                  logits {tuple(logits.shape)}  (predict next char at each step)")
        print(f"greedy generate:  ids {generated[0].tolist()}")
        print(f"                  text {vocab.decode(generated[0].tolist())!r}")

    print("\n=== Vocabulary ===")
    print(f"size:           {vocab.size}")
    print(f"charset chars:  {len(vocab.itos) - 3}")
    print(flush=True)
    return model


def export_sample_pair(cfg: Config, model: OcrVlm) -> None:
    vocab = model.vocab
    rng = random.Random(cfg.data.seed)
    width, height = rng.choice(cfg.data.image_sizes)
    sample = render_text_sample(
        width,
        height,
        rng,
        line_counts=cfg.data.line_counts,
        sentence_len_min=cfg.data.sentence_len_min,
        sentence_len_max=cfg.data.sentence_len_max,
        variations=cfg.data.variations,
    )
    device = training_device()
    model.to(device)
    tensor = image_to_tensor(sample.image).unsqueeze(0).to(device)
    with torch.no_grad():
        generated = model.generate(tensor, max_length=cfg.max_seq_len)
    pred = vocab.decode(generated[0].tolist())
    panel = input_output_row(sample.image, pred)
    gt_panel = text_panel(width, height, sample.text, label="gt")
    output_panel = text_panel(width, height, pred, label="output", fill=(0, 80, 0))

    if _PREVIEW_DIR.exists():
        shutil.rmtree(_PREVIEW_DIR)
    _PREVIEW_DIR.mkdir(parents=True)
    input_path = _PREVIEW_DIR / "input.jpg"
    gt_path = _PREVIEW_DIR / "gt.jpg"
    output_path = _PREVIEW_DIR / "output.jpg"
    panel_path = _PREVIEW_DIR / "preview.jpg"
    sample.image.save(input_path, format="JPEG", quality=95)
    gt_panel.save(gt_path, format="JPEG", quality=95)
    output_panel.save(output_path, format="JPEG", quality=95)
    panel.save(panel_path, format="JPEG", quality=95)

    print("\n=== Sample I/O ===")
    print(f"canvas:       {width}×{height}")
    print(f"gt text:      {sample.text!r}")
    print(f"pred text:    {pred!r}")
    print(f"gt tokens:    {len(vocab.encode(sample.text))}  (incl. BOS/EOS)")
    print(f"pred tokens:  {len(generated[0])}")
    print(f"preview_dir:  {_PREVIEW_DIR}")
    print(f"input:        {input_path}")
    print(f"gt:           {gt_path}")
    print(f"output:       {output_path}")
    print(f"panel:        {panel_path}")


def run(cfg: Config, *, export_sample: bool = True) -> None:
    torch.set_grad_enabled(False)
    model = print_model(cfg, dummy_forward=True)
    print("\n=== Config ===")
    print(f"image_sizes:    {list(cfg.data.image_sizes)}")
    print(f"max_seq_len:    {cfg.max_seq_len}")
    print(f"sentence_len:   {cfg.data.sentence_len_min}–{cfg.data.sentence_len_max}")
    print(f"line_counts:    {list(cfg.data.line_counts)}")
    v = cfg.data.variations
    print(
        "variations:     "
        f"background={v.background} grain={v.grain} ink={v.ink} "
        f"handwritten={v.handwritten} font={v.font} font_size={v.font_size} "
        f"x_offset={v.x_offset} line_spacing={v.line_spacing}"
    )
    print(f"samples/epoch:  {cfg.data.samples_per_epoch:,}")
    if export_sample:
        export_sample_pair(cfg, model)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Print OCR VLM architecture")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--no-sample",
        action="store_true",
        help="Skip exporting a sample input/output pair to _preview/",
    )
    args = parser.parse_args(argv)
    run(load_config(args.config), export_sample=not args.no_sample)


if __name__ == "__main__":
    main()
