"""Print FPN model architecture, parameter counts, and dummy-forward shapes."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from canvas import merge_panels, side_by_side_mask, side_by_side_mask_channel
from config import CONFIG_PATH, Config, datasets_cache_dir, load_config
from data import (
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
_MASK_NAMES = ("walls", "doors", "windows")


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
    "fpn.l5_transformer",
    "fpn.merge_td",
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
    h = cfg.input_size
    sizes = _spatial_sizes(h)
    c2, c3, c4, c5 = cfg.net.backbone_channels
    stem = cfg.net.stem_channels
    depth = cfg.net.depth
    width = cfg.net.fpn_width

    print("\n=== Architecture ===", flush=True)
    print(f"Task:     grayscale floor plan → walls / doors / windows", flush=True)
    print(f"Input:    (1, {h}, {h})   gray → invert (white walls on black)", flush=True)
    print(f"Output:   (3, {h}, {h})   wall / door / window logits", flush=True)
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
    print(f"├─ YoloLikeFPN (width={width})", flush=True)
    print(f"│  ├─ lateral 1×1    c2..c5 → {width} ch", flush=True)
    l5 = cfg.net.l5_transformer
    if l5 is not None:
        print(
            f"│  ├─ L5Transformer  {width}→{l5.dim} ch, depth={l5.depth}, "
            f"heads={l5.heads}, 3D RoPE (y,x,level={l5.level})",
            flush=True,
        )
        print(f"│  └─ top-down      P5→P4→P3→P2", flush=True)
    else:
        print(f"│  └─ top-down      P5→P4→P3→P2", flush=True)
    print(f"└─ FPNMaskHead", flush=True)
    print(
        f"   fuse P2+P3+P4 @ {sizes['c2']}×{sizes['c2']} → upsample 4× → {h}×{h} mask",
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
        size = cfg.input_size
        with torch.no_grad():
            x = torch.zeros(1, 1, size, size)
            features = model.backbone(x)
            pyramid = model.fpn(features)
            out = model(x)
        print("\n=== Dummy forward ===")
        print(f"input:       {tuple(x.shape)}")
        for index, feat in enumerate(features, start=2):
            print(f"c{index}:         {tuple(feat.shape)}")
        print(f"p2:          {tuple(pyramid['p2'].shape)}")
        print(f"p3:          {tuple(pyramid['p3'].shape)}")
        print(f"p4:          {tuple(pyramid['p4'].shape)}")
        print(f"mask_logits: {tuple(out.mask_logits.shape)}")

    print("\n=== Config ===")
    print(f"input_size:   {cfg.input_size}")
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
    row = train_rows[50]
    stem = str(row["stem"])
    image = prepare_image(row["image"], cfg.input_size)
    extraction = scale_extraction(
        json.loads(str(row["extraction"])),
        extraction_scale(cfg.input_size),
    )
    gt_targets = build_floorplan_targets(extraction, cfg.input_size)

    device = training_device()
    model.to(device)
    with torch.no_grad():
        logits = model(image_to_tensor(image).unsqueeze(0).to(device)).mask_logits
    pred = (torch.sigmoid(logits[0]) > 0.5).cpu().numpy()
    input_view = image.convert("RGB")
    pred_view = side_by_side_mask(input_view, pred)
    gt_view = side_by_side_mask(input_view, gt_targets)
    panel = merge_panels([pred_view, gt_view])

    if _PREVIEW_DIR.exists():
        shutil.rmtree(_PREVIEW_DIR)
    _PREVIEW_DIR.mkdir(parents=True)
    input_path = _PREVIEW_DIR / "input.jpg"
    output_path = _PREVIEW_DIR / "output.jpg"
    gt_path = _PREVIEW_DIR / "gt.jpg"
    panel_path = _PREVIEW_DIR / "preview.jpg"
    input_view.save(input_path, format="JPEG", quality=95)
    pred_view.save(output_path, format="JPEG", quality=95)
    gt_view.save(gt_path, format="JPEG", quality=95)
    panel.save(panel_path, format="JPEG", quality=95)

    channel_paths: list[tuple[str, Path]] = []
    for channel, name in enumerate(_MASK_NAMES):
        pred_channel_path = _PREVIEW_DIR / f"pred_{name}.jpg"
        gt_channel_path = _PREVIEW_DIR / f"gt_{name}.jpg"
        side_by_side_mask_channel(input_view, pred, channel).save(
            pred_channel_path, format="JPEG", quality=95
        )
        side_by_side_mask_channel(input_view, gt_targets, channel).save(
            gt_channel_path, format="JPEG", quality=95
        )
        channel_paths.append((f"pred_{name}", pred_channel_path))
        channel_paths.append((f"gt_{name}", gt_channel_path))

    print("\n=== Sample I/O ===")
    print(f"dataset:      {cfg.data.dataset}")
    print(f"stem:         {stem}")
    print(f"preview_dir:  {_PREVIEW_DIR}")
    print(f"input:        {input_path}")
    print(f"output:       {output_path}")
    print(f"gt:           {gt_path}")
    print(f"panel:        {panel_path}")
    for label, path in channel_paths:
        print(f"{label + ':':<14} {path}")


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
