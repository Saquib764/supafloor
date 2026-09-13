"""Print OCR VLM architecture, parameter counts, and dummy-forward shapes."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import torch

from config import CONFIG_PATH, Config, load_config, print_data_schedule
from data import image_to_tensor
from renderer.canvas import input_output_row, text_panel
from renderer.corpus import require_corpus
from renderer.paragraph_render import render_paragraph_sample
from device import training_device
from model import (
    OcrVlm,
    layer_order,
    layer_param_counts,
    print_architecture_summary,
    print_vision_encode_summary,
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
    print(f"  4. Stop at <EOS>  (id {vocab.eos_id})  or max_seq_len={cfg.max_seq_len.final}", flush=True)
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
    print(f"  • lm_head → softmax over {vocab.size} char tokens", flush=True)


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

    layers = layer_param_counts(model)
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

    settings = cfg.data.at_final()
    example_w, example_h = settings.image_sizes[0]
    print_architecture_summary(
        canvas_sizes=cfg.data.all_image_sizes,
        example_width=example_w,
        example_height=example_h,
        vocab_size=vocab.size,
    )
    print_vision_encode_summary(
        canvas_sizes=cfg.data.all_image_sizes,
        example_width=example_w,
        example_height=example_h,
        paragraph_counts=cfg.data.paragraph_counts,
        paragraph_len_max=cfg.data.paragraph_len_max,
        max_seq_len=cfg.max_seq_len.final,
    )
    _print_prediction_summary(cfg, vocab)
    print("\n=== Parameters ===", flush=True)
    for name in layer_order():
        count, size = layers[name]
        print(f"{name + ':':<22} {_fmt_line(count, size)}", flush=True)
    print(f"{'trainable:':<22} {_fmt_line(trainable, trainable_bytes)}", flush=True)
    print(f"{'total:':<22} {_fmt_line(total, total_bytes)}", flush=True)

    if dummy_forward:
        fw_w, fw_h = cfg.data.at_final().image_sizes[0]
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
    corpus = require_corpus(cfg.words.path)
    rng = random.Random(cfg.data.seed)
    settings = cfg.data.at_final()
    width, height = rng.choice(settings.image_sizes)
    sample = render_paragraph_sample(
        width,
        height,
        rng,
        corpus,
        settings.variations,
        paragraph_counts=cfg.data.paragraph_counts,
        paragraph_len_max=cfg.data.paragraph_len_max,
    )
    device = training_device()
    model.to(device)
    tensor = image_to_tensor(sample.image).unsqueeze(0).to(device)
    label_len = len(vocab.encode(sample.text, add_bos_eos=True))
    max_length = min(label_len + 32, cfg.max_seq_len.final)
    with torch.no_grad():
        generated = model.generate(tensor, max_length=max_length)
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
    print_data_schedule(cfg)
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
