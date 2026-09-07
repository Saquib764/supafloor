#!/usr/bin/env python3
"""SAM 2.1 inference worker — run inside sam_trainer venv (transformers >= 4.48)."""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from config import CONFIG_PATH, load_config
from model import load_predictor, predict_point, predict_points


def _send(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _read_image_b64(value: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB")


def _mask_to_b64(mask: np.ndarray) -> str:
    array = (mask.astype(np.uint8) * 255)
    buf = io.BytesIO()
    Image.fromarray(array, mode="L").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description="SAM infer worker for florence infer UI")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()

    cfg = load_config(args.config)
    predictor = load_predictor(cfg)
    _send({"ok": True, "ready": True})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            cmd = req["cmd"]
            if cmd == "shutdown":
                break
            if cmd == "predict_point":
                image = _read_image_b64(req["image_b64"])
                mask, score = predict_point(
                    predictor,
                    image,
                    (float(req["x"]), float(req["y"])),
                    multimask_output=cfg.infer.multimask_output,
                )
                _send({"ok": True, "score": score, "mask_b64": _mask_to_b64(mask)})
                continue
            if cmd == "predict_points":
                image = _read_image_b64(req["image_b64"])
                points = [(float(x), float(y)) for x, y in req["points"]]
                mask, score = predict_points(
                    predictor,
                    image,
                    points,
                    multimask_output=cfg.infer.multimask_output,
                )
                _send({"ok": True, "score": score, "mask_b64": _mask_to_b64(mask)})
                continue
            _send({"ok": False, "error": f"unknown cmd: {cmd}"})
        except Exception as exc:
            _send({"ok": False, "error": str(exc)})


if __name__ == "__main__":
    main()
