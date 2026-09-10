"""Print FPN model architecture, parameter counts, and dummy-forward shapes."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import torch

from canvas import columns_input_masks, stack_rows
from config import CONFIG_PATH, Config, datasets_cache_dir, load_config
from data import (
    augment_sample,
    build_floorplan_targets,
    extraction_scale,
    image_to_tensor,
    load_train_split,
    prepare_image,
    scale_extraction,
)
from device import training_device
from net.model import FloorplanFPN, build_model

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
    "backbone.stem",
    "backbone.stages.0",
    "backbone.stages.1",
    "backbone.stages.2",
    "backbone.stages.3",
    "fpn.lateral",
    "fpn.lateral_stem",
    "fpn.l5_transformer",
    "fpn.context_p4",
    "fpn.context_p3",
    "fpn.merge_td",
    "fpn.p2_to_p1",
    "fpn.merge_p1",
    "mask_head",
)


def _spatial_sizes(input_size: int) -> dict[str, int]:
    return {
        "stem": input_size // 2,
        "c2": input_size // 4,
        "c3": input_size // 8,
        "c4": input_size // 16,
        "c5": input_size // 32,
    }


def _print_architecture_summary(cfg: Config) -> None:
    h = max(cfg.input_sizes)
    sizes = _spatial_sizes(h)
    c2, c3, c4, c5 = cfg.net.backbone_channels
    stem = cfg.net.stem_channels
    depth = cfg.net.depth
    width = cfg.net.fpn_width
    p1 = cfg.net.p1_width
    out = cfg.net.out_channels

    print("\n=== Architecture ===", flush=True)
    print(f"Task:     grayscale floor plan → background / walls / rooms / doors / windows", flush=True)
    print(f"Input:    (1, {h}, {h})   gray → invert (white walls on black)", flush=True)
    print(f"Output:   ({out}, {h}, {h})   bg / wall / room / door / window logits", flush=True)
    print(flush=True)
    print("FloorplanFPN", flush=True)
    print(f"├─ CnnBackbone (depth={depth} residual blocks per stage)", flush=True)
    print(
        f"│  ├─ stem           1→{stem} ch, stride 2   @ {sizes['stem']}×{sizes['stem']}",
        flush=True,
    )
    for index, (label, size, channels) in enumerate(
        (
            ("stage 0 → c2", sizes["c2"], c2),
            ("stage 1 → c3", sizes["c3"], c3),
            ("stage 2 → c4", sizes["c4"], c4),
            ("stage 3 → c5", sizes["c5"], c5),
        ),
        start=0,
    ):
        branch = "│  └─" if index == 3 else "│  ├─"
        print(f"{branch} {label:<16} → {channels} ch @ {size}×{size}", flush=True)
    print(f"├─ YoloLikeFPN (width={width}, p1={p1})", flush=True)
    print(f"│  ├─ lateral 1×1    c2..c5 → {width} ch", flush=True)
    print(f"│  ├─ lateral stem  → P1 {p1} ch @ {sizes['stem']}×{sizes['stem']}", flush=True)
    l5 = cfg.net.l5_transformer
    if l5 is not None:
        print(
            f"│  ├─ L5Transformer  {width}→{l5.dim} ch, depth={l5.depth}, "
            f"heads={l5.heads}, 2D RoPE (y/(H-1), x/(W-1))",
            flush=True,
        )
    print(f"│  ├─ context DW7×7 on P4, P3", flush=True)
    print(f"│  └─ top-down      P5→P4→P3→P2→P1", flush=True)
    print(f"└─ FPNMaskHead", flush=True)
    print(
        f"   fuse P1+P2+P3 @ {sizes['stem']}×{sizes['stem']} → upsample 2× → {h}×{h} mask",
        flush=True,
    )


def _layer_param_counts(model: FloorplanFPN) -> dict[str, tuple[int, int]]:
    counts = {name: (0, 0) for name in _LAYER_ORDER}
    for name, param in model.named_parameters():
        matched = next(
            (key for key in _LAYER_ORDER if name == key or name.startswith(f"{key}.")),
            None,
        )
        if matched is None:
            raise RuntimeError(f"parameter not in known layers: {name}")
        count, size = counts[matched]
        n = param.numel()
        counts[matched] = (count + n, size + n * param.element_size())
    if model.fpn.l5_transformer is None:
        del counts["fpn.l5_transformer"]
    return counts


def print_model(cfg: Config, *, dummy_forward: bool = True) -> FloorplanFPN:
    model = build_model(cfg)
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

    _print_architecture_summary(cfg)
    print("\n=== Parameters ===", flush=True)
    for name in _LAYER_ORDER:
        if name not in layers:
            continue
        count, size = layers[name]
        print(f"{name + ':':<22} {_fmt_line(count, size)}", flush=True)
    print(f"{'trainable:':<22} {_fmt_line(trainable, trainable_bytes)}", flush=True)
    print(f"{'total:':<22} {_fmt_line(total, total_bytes)}", flush=True)

    if dummy_forward:
        size = max(cfg.input_sizes)
        with torch.no_grad():
            x = torch.zeros(1, 1, size, size)
            stem, features = model.backbone(x)
            pyramid = model.fpn(stem, features)
            out = model(x)
        print("\n=== Dummy forward ===")
        print(f"input:       {tuple(x.shape)}")
        for index, feat in enumerate(features, start=2):
            print(f"c{index}:         {tuple(feat.shape)}")
        print(f"p1:          {tuple(pyramid['p1'].shape)}")
        print(f"p2:          {tuple(pyramid['p2'].shape)}")
        print(f"p3:          {tuple(pyramid['p3'].shape)}")
        print(f"p4:          {tuple(pyramid['p4'].shape)}")
        print(f"mask_logits: {tuple(out.mask_logits.shape)}")

    print("\n=== Config ===")
    print(f"input_sizes:  {list(cfg.input_sizes)}")
    print(f"fpn_width:    {cfg.net.fpn_width}")
    print(f"depth:        {cfg.net.depth}")
    print(flush=True)
    return model


def export_sample_pair(cfg: Config, model: FloorplanFPN) -> None:
    cache_dir = datasets_cache_dir(cfg)
    print(f"Loading sample row from {cfg.data.dataset} …")
    train_rows, _train_count, _test_count = load_train_split(
        cfg.data.dataset,
        test_fraction=cfg.data.test_fraction,
        revision=cfg.data.revision,
        cache_dir=cache_dir,
    )
    row = train_rows[90]
    stem = str(row["stem"])
    size = max(cfg.input_sizes)
    image = prepare_image(row["image"], size)
    extraction = scale_extraction(
        json.loads(str(row["extraction"])),
        extraction_scale(size),
    )
    gt_targets = build_floorplan_targets(extraction, size, cfg.mask_channels)
    rng = random.Random(cfg.data.seed)
    image, gt_targets = augment_sample(
        image,
        gt_targets,
        rng=rng,
        augment=cfg.data.augment,
        background_channel=cfg.mask_channels.background,
    )
    mask_labels = cfg.mask_channels.labels()

    device = training_device()
    model.to(device)
    with torch.no_grad():
        logits = model(image_to_tensor(image).unsqueeze(0).to(device)).mask_logits
    pred = (torch.sigmoid(logits[0]) > 0.5).cpu().numpy()
    gt_row = columns_input_masks(
        image, gt_targets, mask_labels=mask_labels, background_channel=cfg.mask_channels.background
    )
    pred_row = columns_input_masks(
        image, pred, mask_labels=mask_labels, background_channel=cfg.mask_channels.background
    )
    panel = stack_rows([gt_row, pred_row])

    if _PREVIEW_DIR.exists():
        shutil.rmtree(_PREVIEW_DIR)
    _PREVIEW_DIR.mkdir(parents=True)
    input_path = _PREVIEW_DIR / "input.jpg"
    gt_path = _PREVIEW_DIR / "gt.jpg"
    output_path = _PREVIEW_DIR / "output.jpg"
    panel_path = _PREVIEW_DIR / "preview.jpg"
    image.save(input_path, format="JPEG", quality=95)
    gt_row.save(gt_path, format="JPEG", quality=95)
    pred_row.save(output_path, format="JPEG", quality=95)
    panel.save(panel_path, format="JPEG", quality=95)

    print("\n=== Sample I/O ===")
    print(f"dataset:      {cfg.data.dataset}")
    print(f"stem:         {stem}")
    print(f"preview_dir:  {_PREVIEW_DIR}")
    print(f"input:        {input_path}")
    print(f"gt:           {gt_path}")
    print(f"output:       {output_path}")
    print(f"panel:        {panel_path}")


def run(cfg: Config, *, export_sample: bool = True) -> None:
    torch.set_grad_enabled(False)
    model = print_model(cfg)
    if export_sample:
        export_sample_pair(cfg, model)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Print floor-plan FPN model architecture")
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
