"""Multi-resolution FPN room and structure mask inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import CONFIG_PATH, load_config
from util.fpn_pipeline import run_fpn_pipeline

_ROOT = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="FPN room and structure masks at 512, 768, and 1024 px",
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--image",
        type=Path,
        default=_ROOT.parent / "fpn_trainer" / "test_image" / "002.jpeg",
        help="Floor plan image path",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    outputs = run_fpn_pipeline(cfg, args.image)
    for step, paths in outputs.items():
        if step == "floorplan_region":
            region_paths = paths
            print(f"floorplan region → {region_paths['panel']}")
            print(f"floorplan region mask → {region_paths['mask']}")
            print(f"floorplan polygon → {region_paths['json']}")
            print(f"masked canvas → {region_paths['masked_canvas']}")
            continue
        if step == "rooms":
            print(f"rooms markdown → {paths['markdown']}")
            print(f"room mask overlays → {paths['overlays']}")
            print(f"wall input → {paths['input']}")
            print(f"wall based room mask final → {paths['wall_based_room_mask_final_root']}")
            continue
        if step == "area_masks":
            print(f"open area mask → {paths['open_area']}")
            print(f"open area overlay → {paths['open_area_overlay']}")
            print(f"wall area mask → {paths['wall_area']}")
            continue
        room_path, structure_path = paths
        print(f"{step} room masks → {room_path}")
        print(f"{step} structure masks → {structure_path}")


if __name__ == "__main__":
    main()
