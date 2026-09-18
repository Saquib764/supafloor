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
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Override fpn checkpoint path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override fpn output base directory (per-image stem dirs are created beneath it)",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    outputs = run_fpn_pipeline(
        cfg,
        args.image,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
    )
    for step, paths in outputs.items():
        if step not in ("step-001", "step-002"):
            print(f"{step} → {paths}")
            continue
        print(f"{step}/ → {paths['dir']}/")
        if step == "step-001":
            print(f"  floorplan region mask → {paths['mask']}")
            print(f"  floorplan polygon → {paths['json']}")
        if step == "step-002":
            print(f"  input cropped → {paths['input_cropped']}")
        print(f"  input masked → {paths['input_masked']}")
        print(f"  raw masks → {paths['masks']}")
        print(f"  cleaned masks → {paths['masks_clean']}")
        final_masks = paths["final_masks"]
        print(f"  final masks → {final_masks['background'].parent}/")
        for layer, mask_path in final_masks.items():
            if layer == "polygons":
                continue
            print(f"    {layer} → {mask_path}")
        polygon_paths = final_masks["polygons"]
        print(f"  final mask polygons → {polygon_paths['room'].parent}/")
        for layer, json_path in polygon_paths.items():
            print(f"    {layer} → {json_path}")
        rooms = paths["rooms"]
        print(f"  rooms markdown → {rooms['markdown']}")
        print(f"  room mask overlays → {rooms['overlays']}")
        print(f"  wall input → {rooms['input']}")
        structure = paths["structure_polygons"]
        print(f"  structure polygons → {structure['dir']}/")
        print(f"    wall centerline → {structure['wall_centerline']}")
        print(f"    opencv thinning → {structure['opencv_thinning']}")
        print(f"    skimage skeleton → {structure['skimage_skeleton']}")
        print(f"    structure polylines → {structure['polylines_json']}")


if __name__ == "__main__":
    main()
