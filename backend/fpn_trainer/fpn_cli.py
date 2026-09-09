"""CLI: uv run fpn-train train|inspect|infer."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import CONFIG_PATH, load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train star-prompted room segmentation (CNN + FPN)"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train on Hub floor-plan rows")
    p_train.add_argument("--config", type=Path, default=CONFIG_PATH)

    p_inspect = sub.add_parser("inspect", help="Print model architecture and dummy-forward shapes")
    p_inspect.add_argument("--config", type=Path, default=CONFIG_PATH)

    p_infer = sub.add_parser("infer", help="Segment the room marked by a star")
    p_infer.add_argument("--config", type=Path, default=CONFIG_PATH)
    p_infer.add_argument("--image", type=Path, required=True)
    p_infer.add_argument("--cx", type=float, required=True)
    p_infer.add_argument("--cy", type=float, required=True)
    p_infer.add_argument("--radius", type=float, default=None)
    p_infer.add_argument("--checkpoint", type=Path, default=None)
    p_infer.add_argument("--output", type=Path, default=None)

    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    if args.cmd == "inspect":
        from preview import run as run_preview

        run_preview(cfg)
        return

    if args.cmd == "train":
        from train import run

        run(cfg)
        return

    from PIL import Image

    from canvas import merge_panels, overlay_masks
    from infer import _device, infer_mask, load_model

    device = _device()
    checkpoint = args.checkpoint or cfg.infer.checkpoint
    model = load_model(cfg, checkpoint, device)
    image = Path(args.image)
    radius = args.radius if args.radius is not None else (cfg.star.min_radius + cfg.star.max_radius) / 2
    starred, mask = infer_mask(
        model,
        Image.open(image),
        cx=args.cx,
        cy=args.cy,
        radius=radius,
        cfg=cfg,
        device=device,
    )
    panel = merge_panels([starred, overlay_masks(starred, mask)])
    out = args.output or cfg.train.session_dir / "infer" / f"{image.stem}_fpn.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out, format="JPEG", quality=95)
    print(f"infer → {out}")


if __name__ == "__main__":
    main()
