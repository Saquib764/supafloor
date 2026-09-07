"""CLI: uv run sam-train train|inspect."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import CONFIG_PATH, load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="SAM 2.1 floor-plan segmentation")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Fine-tune on floorplan-to-json point prompts")
    p_train.add_argument("--config", type=Path, default=CONFIG_PATH)

    p_inspect = sub.add_parser("inspect", help="Save dataset samples and print model architecture")
    p_inspect.add_argument("--config", type=Path, default=CONFIG_PATH)
    p_inspect.add_argument("--count", type=int, default=1, help="Random samples to save (0 = model only)")
    p_inspect.add_argument("--seed", type=int, default=None, help="RNG seed for sample selection")

    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    if args.cmd == "train":
        from train import run

        run(cfg)
        return

    if args.cmd == "inspect":
        from preview import run as run_preview

        run_preview(cfg, count=args.count, seed=args.seed)
        return


if __name__ == "__main__":
    main()
