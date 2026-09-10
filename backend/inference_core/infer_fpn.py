"""Multi-resolution fpn_five structure mask inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import CONFIG_PATH, load_config
from util.fpn_pipeline import run_fpn_pipeline

_ROOT = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="FPN Five masks (background/rooms/walls/doors/windows) at 512, 768, and 1024 px",
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--image",
        type=Path,
        default=_ROOT / "test_image" / "fp01.jpg",
        help="Floor plan image path",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    outputs = run_fpn_pipeline(cfg, args.image)
    for step, paths in outputs.items():
        if step == "floorplan_region":
            region_paths = paths
            print(f"floorplan region mask → {region_paths['mask']}")
            print(f"floorplan polygon → {region_paths['json']}")
            print(f"masked canvas → {region_paths['masked_canvas']}")
            continue
        if step == "rooms":
            print(f"rooms markdown → {paths['markdown']}")
            print(f"room mask overlays → {paths['overlays']}")
            print(f"wall input → {paths['input']}")
            continue
        print(f"{step} masks → {paths}")


if __name__ == "__main__":
    main()
