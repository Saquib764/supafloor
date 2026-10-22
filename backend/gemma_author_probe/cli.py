"""CLI: uv run gemma-author-probe train"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
from pathlib import Path

from config import CONFIG_PATH, load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Frozen Gemma 3 270M + probe token authorship classifier (18 classes)"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_preview = sub.add_parser("preview", help="Summarize model, head, and parameter counts")
    p_preview.add_argument("--config", type=Path, default=CONFIG_PATH)
    p_preview.add_argument(
        "--config-only",
        action="store_true",
        help="Skip hub download; print config/head summary only",
    )

    p_train = sub.add_parser("train", help="Train probe, head, and Gemma blocks on train.csv")
    p_train.add_argument("--config", type=Path, default=CONFIG_PATH)
    p_train.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Stop after this many optimizer steps (smoke test). Default: config train.max_steps",
    )

    p_run = sub.add_parser("run-test", help="Eval held-out prompts from the saved checkpoint")
    p_run.add_argument("--config", type=Path, default=CONFIG_PATH)
    p_run.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    if args.cmd == "preview":
        from preview import run as run_preview

        run_preview(cfg, build_from_hub=not args.config_only)
        return

    if args.cmd == "train":
        from train import run

        run(cfg, max_steps=args.max_steps)
        return

    if args.cmd == "run-test":
        from run_test import run_test

        run_test(cfg, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
