"""CLI: uv run modern-bert-decision inspect|train|infer|..."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import CONFIG_PATH, load_config


def _add_infer_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint directory (default: infer.checkpoint in config.yaml)",
    )
    parser.add_argument(
        "--request",
        type=Path,
        default=None,
        help="JSON request file (default: example_request.json)",
    )
    parser.add_argument(
        "--from-dataset",
        action="store_true",
        help="Use one row from the configured dataset instead of a JSON request file",
    )
    parser.add_argument("--from-race", action="store_true", help=argparse.SUPPRESS)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="ModernBERT encoder decision model (multi-option scoring, no generation)"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_inspect = sub.add_parser("inspect", help="Print architecture and dummy-forward tensor shapes")
    p_inspect.add_argument("--config", type=Path, default=CONFIG_PATH)
    p_inspect.add_argument(
        "--no-samples",
        action="store_true",
        help="Skip printing sample training rows from the configured dataset",
    )
    p_inspect.add_argument("--sample-count", type=int, default=2)

    p_verify = sub.add_parser("verify", help="Run hand-written architecture checks")
    p_verify.add_argument("--config", type=Path, default=CONFIG_PATH)

    p_train = sub.add_parser("train", help="Train on data.dataset (dataset_manager registry)")
    p_train.add_argument("--config", type=Path, default=CONFIG_PATH)
    p_train.add_argument(
        "--test-mode",
        action="store_true",
        help="Train on a fixed subset (train.test_mode.samples) instead of the full train split",
    )
    p_train.add_argument(
        "--test-samples",
        type=int,
        default=None,
        metavar="N",
        help="Override train.test_mode.samples (implies --test-mode)",
    )
    p_train.add_argument(
        "--test-refresh",
        action="store_true",
        help="Resample test_train_samples.jsonl (delete cached fixture first)",
    )

    p_dl = sub.add_parser(
        "download-data",
        help="Download configured dataset from Hub into its local_root JSONL splits",
    )
    p_dl.add_argument("--config", type=Path, default=CONFIG_PATH)
    p_dl.add_argument("--force", action="store_true", help="Re-download even if marker exists")

    p_infer = sub.add_parser(
        "infer",
        help="Score options using infer.checkpoint (default: training_session/last)",
    )
    _add_infer_args(p_infer)

    p_predict = sub.add_parser("predict", help="Alias for infer")
    _add_infer_args(p_predict)

    p_masked = sub.add_parser(
        "masked-infer",
        help="Vanilla ModernBERT MLM: random N rows from configured dataset",
    )
    p_masked.add_argument("--config", type=Path, default=CONFIG_PATH)
    p_masked.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Random row count (default: 20)",
    )
    p_masked.add_argument(
        "--split",
        type=str,
        default=None,
        help="Split name (default: data.val_split)",
    )
    p_masked.add_argument("--seed", type=int, default=None, help="Row selection seed")
    p_masked.add_argument("--quiet", action="store_true", help="Accuracy summary only")

    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    if args.cmd == "inspect":
        from preview import run as run_preview

        run_preview(
            cfg,
            show_samples=not args.no_samples,
            sample_count=max(1, args.sample_count),
        )
        return

    if args.cmd == "verify":
        from verify import run_checks

        run_checks(cfg)
        return

    if args.cmd == "train":
        from dataclasses import replace

        from train import run as run_train

        train = cfg.train
        train_updates: dict[str, object] = {}
        if args.test_mode:
            train_updates["test_mode"] = True
        if args.test_samples is not None:
            train_updates["test_samples"] = max(1, int(args.test_samples))
            train_updates["test_mode"] = True
        if train_updates:
            cfg = replace(cfg, train=replace(train, **train_updates))
        run_train(cfg, test_refresh=bool(args.test_refresh))
        return

    if args.cmd == "download-data":
        from data import download_dataset_local

        path = download_dataset_local(cfg, force=args.force)
        print(f"{cfg.data.dataset} local data → {path}")
        return

    if args.cmd == "masked-infer":
        from masked_infer import run_masked_infer

        run_masked_infer(
            cfg,
            samples=max(1, args.samples),
            split=args.split,
            seed=args.seed,
            verbose=not args.quiet,
        )
        return

    from infer import run_infer

    checkpoint = args.checkpoint or cfg.infer.checkpoint
    run_infer(
        cfg,
        checkpoint=checkpoint,
        request_path=args.request,
        from_dataset=bool(getattr(args, "from_dataset", False) or getattr(args, "from_race", False)),
    )


if __name__ == "__main__":
    main()
