"""CLI: uv run unet-train train|inspect."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import CONFIG_PATH, load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a frozen-taef2 + ViT floor-plan reconstructor")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train on Hub image → simple_image pairs")
    p_train.add_argument("--config", type=Path, default=CONFIG_PATH)

    p_inspect = sub.add_parser("inspect", help="Print model architecture and dummy-forward shapes")
    p_inspect.add_argument("--config", type=Path, default=CONFIG_PATH)

    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    if args.cmd == "inspect":
        from preview import run as run_preview

        run_preview(cfg)
        return

    from train import run

    run(cfg)


if __name__ == "__main__":
    main()
