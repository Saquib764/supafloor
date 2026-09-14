"""CLI: uv run vlm-ocr-train train|inspect|holdout|benchmark|infer."""

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

    p_holdout = sub.add_parser("holdout", help="Generate holdout dataset")
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

    p_infer = sub.add_parser("infer", help="Run OCR on rendered test_samples/")
    p_infer.add_argument("--config", type=Path, default=CONFIG_PATH)
    p_infer.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path (default: infer.checkpoint from config)",
    )
    p_infer.add_argument(
        "--samples",
        type=Path,
        default=None,
        help="Sample definitions JSON (default: test_samples/samples.json)",
    )
    p_infer.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for preview panels (default: training_session/infer)",
    )
    p_infer.add_argument(
        "--width",
        type=int,
        default=None,
        help="Override canvas width in pixels (default: width from samples JSON)",
    )
    p_infer.add_argument(
        "--height",
        type=int,
        default=None,
        help="Override canvas height in pixels (default: height from samples JSON)",
    )
    p_infer.add_argument(
        "--reseed",
        action="store_true",
        help="Ignore saved seeds and generate new random render variations",
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

    if args.cmd == "infer":
        from infer import run as run_infer

        run_infer(
            cfg,
            checkpoint=args.checkpoint,
            samples_path=args.samples,
            output_dir=args.output_dir,
            canvas_size=(
                (
                    args.width if args.width is not None else 256,
                    args.height if args.height is not None else 256,
                )
                if args.width is not None or args.height is not None
                else None
            ),
            reseed=args.reseed,
        )
        return

    from train import run

    run(cfg)


if __name__ == "__main__":
    main()
