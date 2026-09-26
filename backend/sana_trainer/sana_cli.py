"""CLI: uv run sana-train train|preview."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from config import CONFIG_PATH, load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Sana 0.6B (LoRA or full transformer)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train on image+caption Hub or local dataset")
    p_train.add_argument("--config", type=Path, default=CONFIG_PATH)
    p_train.add_argument(
        "--full-finetune",
        action="store_true",
        help="Override config: train full transformer weights (VAE and text encoder stay frozen)",
    )
    p_train.add_argument(
        "--from-adapter",
        type=Path,
        default=None,
        help="Continue LoRA training from an adapter directory",
    )

    p_preview = sub.add_parser("preview", help="Print model/LoRA stats and optional sample image")
    p_preview.add_argument("--config", type=Path, default=CONFIG_PATH)
    p_preview.add_argument("--skip-image", action="store_true")

    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    if args.cmd == "preview":
        from preview import run as run_preview

        run_preview(cfg, skip_image=args.skip_image)
        return

    if args.full_finetune:
        cfg = replace(cfg, train=replace(cfg.train, full_finetune=True))
    if args.from_adapter is not None:
        cfg = replace(cfg, train=replace(cfg.train, from_adapter=args.from_adapter))

    from train import run

    run(cfg)


if __name__ == "__main__":
    main()
