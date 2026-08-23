"""CLI: uv run florence-train train|inspect."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import CONFIG_PATH, load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tune Florence-2 for floor-plan rooms")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Fine-tune LoRA on image-to-floorplan pairs")
    p_train.add_argument("--config", type=Path, default=CONFIG_PATH)

    p_inspect = sub.add_parser("inspect", help="Print model, LoRA, and random Hub samples")
    p_inspect.add_argument("--config", type=Path, default=CONFIG_PATH)
    p_inspect.add_argument("--count", type=int, default=3, help="Random samples to print")
    p_inspect.add_argument("--seed", type=int, default=None, help="RNG seed for sample selection")

    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    if args.cmd == "inspect":
        from preview import run as run_preview

        run_preview(cfg, count=args.count, seed=args.seed)
        return

    from train import run

    run(cfg)


if __name__ == "__main__":
    main()
