"""CLI: uv run vlm-ocr-train train|inspect|holdout|benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import CONFIG_PATH, load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a ~1M-param OCR VLM")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train on synthetic text images")
    p_train.add_argument("--config", type=Path, default=CONFIG_PATH)

    p_inspect = sub.add_parser("inspect", help="Print model architecture and parameter count")
    p_inspect.add_argument("--config", type=Path, default=CONFIG_PATH)
    p_inspect.add_argument(
        "--no-sample",
        action="store_true",
        help="Skip exporting a sample input/output pair to _preview/",
    )

    p_holdout = sub.add_parser("holdout", help="Generate the holdout benchmark dataset")
    p_holdout.add_argument("--config", type=Path, default=CONFIG_PATH)
    p_holdout.add_argument(
        "--force",
        action="store_true",
        help="Regenerate holdout even if manifest.jsonl already exists",
    )

    p_benchmark = sub.add_parser("benchmark", help="Run holdout benchmark on a checkpoint")
    p_benchmark.add_argument("--config", type=Path, default=CONFIG_PATH)
    p_benchmark.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path (default: infer.checkpoint from config)",
    )

    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    if args.cmd == "inspect":
        from preview import run as run_preview

        run_preview(cfg, export_sample=not args.no_sample)
        return

    if args.cmd == "holdout":
        from holdout import generate_holdout

        generate_holdout(cfg, force=args.force)
        return

    if args.cmd == "benchmark":
        from benchmark import run as run_benchmark_cmd

        run_benchmark_cmd(cfg, checkpoint=args.checkpoint)
        return

    from train import run

    run(cfg)


if __name__ == "__main__":
    main()
