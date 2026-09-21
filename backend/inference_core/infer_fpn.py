"""FPN Five structure mask inference at 768px."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import CONFIG_PATH, load_config
from util.fpn_pipeline import run_fpn_pipeline

_ROOT = Path(__file__).resolve().parent


def _print_step_outputs(step: str, paths: dict[str, object]) -> None:
    print(f"{step}/ → {paths['dir']}/")
    if step == "step-001":
        print(f"  floorplan region mask → {paths['mask']}")
        print(f"  floorplan polygon → {paths['json']}")
        print(f"  input masked → {paths['input_masked']}")
        print(f"  raw masks → {paths['masks']}")
        print(f"  cleaned masks → {paths['masks_clean']}")
        print(f"  wall mask → {paths['wall_mask']}")
    elif step == "step-002":
        print(f"  input cropped → {paths['input_cropped']}")
        print(f"  input masked → {paths['input_masked']}")
        print(f"  raw masks → {paths['masks']}")
        print(f"  cleaned masks → {paths['masks_clean']}")
        print(f"  wall mask → {paths['wall_mask']}")
    elif step == "step-003":
        print(f"  input masked → {paths['input_masked']}")
        print(f"  input 1024 → {paths['input_1024']}")
        print(f"  room mask 1024 → {paths['room_mask_1024']}")
        print(f"  room polygons render → {paths['room_polygons']}")
        print(f"  room polygons svg → {paths['room_polygons_svg']}")
        print(f"  floorplan llm json → {paths['floorplan_1024_json']}")
        usage = paths["llm_usage"]
        print(
            "  llm tokens → "
            f"input: {usage['input_tokens']:,}  "
            f"cached: {usage.get('cached_tokens', 0):,}  "
            f"cache write: {usage.get('cache_write_tokens', 0):,}  "
            f"output: {usage['output_tokens']:,}  "
            f"total: {usage['total_tokens']:,}"
        )
        print(f"  llm cost → ${usage['cost_usd']:.4f} USD")

    if step in ("step-001", "step-002"):
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
        if "input" in rooms:
            print(f"  wall input → {rooms['input']}")
        structure = paths["structure_polygons"]
        print(f"  structure polygons → {structure['dir']}/")
        print(f"    wall centerline → {structure['wall_centerline']}")
        print(f"    opencv thinning → {structure['opencv_thinning']}")
        print(f"    skimage skeleton → {structure['skimage_skeleton']}")
        print(f"    structure polylines → {structure['polylines_json']}")
        print(f"    hough lines → {structure['hough_lines']}")
    elif step == "step-003":
        final_masks = paths["final_masks"]
        print(f"  final masks → {final_masks['room'].parent}/")
        polygon_paths = final_masks["polygons"]
        for layer in ("room", "door", "window", "background", "wall"):
            if layer in final_masks:
                print(f"    {layer} → {final_masks[layer]}")
            if layer in polygon_paths:
                print(f"    {layer} json → {polygon_paths[layer]}")
        if "input" in final_masks:
            print(f"    input → {final_masks['input']}")
        if "polygon_overlay" in final_masks:
            print(f"    polygon overlay → {final_masks['polygon_overlay']}")
        rooms = paths["rooms"]
        print(f"  rooms markdown → {rooms['markdown']}")
        print(f"  room mask overlays → {rooms['overlays']}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="FPN Five masks (background/rooms/walls/doors/windows) at 768px",
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
    parser.add_argument(
        "--skip-step-2",
        action="store_true",
        help="Skip step-002 crop re-infer; run step-003 on the step-001 canvas",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    outputs = run_fpn_pipeline(
        cfg,
        args.image,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        skip_step_2=args.skip_step_2,
    )
    _print_step_outputs("step-001", outputs["step-001"])
    step_two = outputs.get("step-002")
    if step_two is not None:
        _print_step_outputs("step-002", step_two)
    _print_step_outputs("step-003", outputs["step-003"])


if __name__ == "__main__":
    main()
