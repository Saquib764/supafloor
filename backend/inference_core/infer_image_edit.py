"""Replicate image-edit cleaning + white-region room mask discovery."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import CONFIG_PATH, load_config
from util.image_edit_pipeline import run_image_edit_pipeline
from util.mask_discovery_loop import run_mask_discovery_loop

_ROOT = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Replicate image-edit clean + room mask discovery")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--image",
        type=Path,
        default=_ROOT / "test_image" / "002.jpeg",
        help="Floor plan image path",
    )
    parser.add_argument(
        "--skip-replicate",
        action="store_true",
        help="Skip Replicate calls; use saved 03-no-furniture.png",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    cleaned, run_dir = run_image_edit_pipeline(cfg, args.image, skip_replicate=args.skip_replicate)
    run_mask_discovery_loop(cfg, source_image=cleaned, output_dir=run_dir)


if __name__ == "__main__":
    main()
