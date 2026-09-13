"""CLI: uv run vlm-ocr-train train|inspect|words|holdout|benchmark|infer."""

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

    p_words = sub.add_parser("words", help="Generate word-collection JSON")
    p_words.add_argument("--config", type=Path, default=CONFIG_PATH)
    p_words.add_argument(
        "--force",
        action="store_true",
        help="Regenerate words.json even if it already exists",
    )

    p_holdout = sub.add_parser(
        "holdout",
        help="Generate paragraph holdout dataset from words.json",
    )
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

    p_infer = sub.add_parser("infer", help="Run OCR on images in test_images/")
    p_infer.add_argument("--config", type=Path, default=CONFIG_PATH)
    p_infer.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path (default: infer.checkpoint from config)",
    )
    p_infer.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory of input images (default: test_images/)",
    )
    p_infer.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for prepared inputs and previews (default: training_session/infer)",
    )
    p_infer.add_argument(
        "--canvas-size",
        type=int,
        default=256,
        help="Square white canvas size in pixels (default: 256)",
    )

    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    if args.cmd == "inspect":
        from preview import run as run_preview

        run_preview(cfg, export_sample=not args.no_sample)
        return

    if args.cmd == "words":
        from renderer.words_gen import generate_word_collections

        generate_word_collections(cfg, force=args.force)
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
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            canvas_size=args.canvas_size,
        )
        return

    from train import run

    run(cfg)


if __name__ == "__main__":
    main()
