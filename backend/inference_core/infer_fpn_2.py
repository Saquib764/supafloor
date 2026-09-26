"""FPN Five structure mask inference — step-001 only (no crop re-infer, no LLM step).

Usage (from backend/inference_core):

    uv run python infer_fpn_2.py --image test_image/BE_01.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config import CONFIG_PATH, load_config
from util.fpn_pipeline import run_fpn_pipeline

_ROOT = Path(__file__).resolve().parent


def _print_step_outputs(paths: dict[str, object]) -> None:
    print(f"step-001/ → {paths['dir']}/")
    print(f"  raw masks → {paths['masks']}")
    print(f"  cleaned masks → {paths['masks_clean']}")
    print(f"  wall mask → {paths['wall_mask']}")

    final_masks = paths["final_masks"]
    print(f"  final masks → {final_masks['background'].parent}/")
    for layer, mask_path in final_masks.items():
        if layer in ("polygons", "polygon_overlay"):
            continue
        print(f"    {layer} → {mask_path}")
    if "polygon_overlay" in final_masks:
        print(f"  polygon overlay → {final_masks['polygon_overlay']}")
    polygon_paths = final_masks["polygons"]
    print(f"  final mask polygons → {polygon_paths['room'].parent}/")
    for layer, json_path in polygon_paths.items():
        print(f"    {layer} → {json_path}")
    rooms = paths["rooms"]
    print(f"  rooms markdown → {rooms['markdown']}")
    print(f"  room mask overlays → {rooms['overlays']}")
    print(f"  mask vs polygon → {paths['mask_polygon_comparison']}")
    if "input" in rooms:
        print(f"  wall input → {rooms['input']}")
    if "room_names" in paths:
        room_names = paths["room_names"]
        print(f"  room names json → {room_names['json']}")
        usage = room_names["usage"]
        if isinstance(usage, dict) and "input_tokens" in usage:
            print(
                "  room names llm usage → "
                f"input: {usage['input_tokens']:,}  "
                f"output: {usage['output_tokens']:,}  "
                f"cost: ${usage['cost_usd']:.4f}"
            )
        elif isinstance(usage, dict):
            print(
                "  room names paddle usage → "
                f"detections: {usage.get('detections', 0)}  "
                f"kept: {usage.get('kept', 0)}  "
                f"elapsed: {usage.get('elapsed_ms', 0)}ms"
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="FPN Five masks at step-001 only (skip step-002 and step-003)",
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
        skip_step_2=True,
        skip_step_3=True,
    )
    _print_step_outputs(outputs["step-001"])


if __name__ == "__main__":
    main()
