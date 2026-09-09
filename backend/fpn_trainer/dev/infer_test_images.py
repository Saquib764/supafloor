"""Run fpn_trainer inference on test_image/."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from canvas import merge_panels, overlay_masks
from config import load_config
from infer import _device, infer_mask, load_model

_TEST_DIR = _ROOT / "test_image"
_OUT_DIR = _ROOT / "training_session" / "test_infer"


def main() -> None:
    cfg = load_config()
    device = _device()
    checkpoint = cfg.infer.checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    model = load_model(cfg, checkpoint, device)
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(
        path
        for path in _TEST_DIR.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    for path in paths:
        prepared, mask = infer_mask(
            model,
            Image.open(path),
            cfg=cfg,
            device=device,
        )
        input_view = prepared.convert("RGB")
        panel = merge_panels([input_view, overlay_masks(input_view, mask)])
        out = _OUT_DIR / f"{path.stem}_fpn.jpg"
        panel.save(out, format="JPEG", quality=95)
        print(f"{path.name} → {out}")


if __name__ == "__main__":
    main()
