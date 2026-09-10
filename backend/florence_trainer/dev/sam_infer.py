"""Run sam_trainer inference in its own venv (needs transformers >= 4.48)."""

from __future__ import annotations

import base64
import importlib.util
import io
import json
import subprocess
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

_SAM_ROOT = Path(__file__).resolve().parents[2] / "sam_trainer"
_WORKER = Path(__file__).resolve().parent / "sam_worker.py"
_ISOLATED = ("config", "canvas")


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_sam_modules() -> tuple[Any, Any]:
    saved = {name: sys.modules.get(name) for name in _ISOLATED}
    sam_str = str(_SAM_ROOT.resolve())
    if sam_str not in sys.path:
        sys.path.insert(0, sam_str)
    try:
        sam_config = _load_module("sam_trainer.config", _SAM_ROOT / "config.py")
        sam_canvas = _load_module("sam_trainer.canvas", _SAM_ROOT / "canvas.py")
        return sam_config, sam_canvas
    finally:
        for name, module in saved.items():
            if module is not None:
                sys.modules[name] = module
            else:
                sys.modules.pop(name, None)


def _image_to_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _mask_from_b64(value: str) -> np.ndarray:
    array = np.array(Image.open(io.BytesIO(base64.b64decode(value))).convert("L"))
    return array > 0


@dataclass
class SamRuntime:
    cfg: Any
    load_viewer_image: Any
    apply_masks: Any
    fit_to_canvas: Any
    _proc: subprocess.Popen[str] = field(repr=False)
    _config_path: Path = field(repr=False)

    @classmethod
    def load(cls, config_path: Path | None = None) -> SamRuntime:
        sam_config, sam_canvas = _load_sam_modules()
        path = config_path or sam_config.CONFIG_PATH
        cfg = sam_config.load_config(path)

        proc = subprocess.Popen(
            [
                "uv",
                "run",
                "python",
                str(_WORKER.resolve()),
                "--config",
                str(path.resolve()),
            ],
            cwd=str(_SAM_ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if proc.stdout is None or proc.stdin is None:
            proc.kill()
            raise RuntimeError("Failed to start SAM worker")

        ready_line = proc.stdout.readline()
        if proc.poll() is not None:
            stderr = proc.stderr.read() if proc.stderr else ""
            raise RuntimeError(f"SAM worker exited during startup: {stderr.strip()}")

        ready = json.loads(ready_line)
        if not ready.get("ok"):
            raise RuntimeError(ready.get("error", "SAM worker failed to start"))

        return cls(
            cfg=cfg,
            load_viewer_image=sam_canvas.load_viewer_image,
            apply_masks=sam_canvas.apply_masks,
            fit_to_canvas=sam_canvas.fit_to_canvas,
            _proc=proc,
            _config_path=path.resolve(),
        )

    def _request(self, payload: dict) -> dict:
        if self._proc.stdin is None or self._proc.stdout is None:
            raise RuntimeError("SAM worker is not running")
        if self._proc.poll() is not None:
            stderr = self._proc.stderr.read() if self._proc.stderr else ""
            raise RuntimeError(f"SAM worker exited: {stderr.strip()}")

        self._proc.stdin.write(json.dumps(payload) + "\n")
        self._proc.stdin.flush()
        response = json.loads(self._proc.stdout.readline())
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "SAM inference failed"))
        return response

    def close(self) -> None:
        if self._proc.poll() is not None:
            return
        try:
            if self._proc.stdin is not None:
                self._proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                self._proc.stdin.flush()
        finally:
            self._proc.wait(timeout=30)

    def overlay(
        self,
        image: Image.Image,
        masks: list[np.ndarray],
        points: list[tuple[float, float]],
        *,
        negative_points: list[tuple[float, float]] | None = None,
    ) -> Image.Image:
        return self.apply_masks(image, masks, points, negative_points=negative_points)

    def predict_point(
        self,
        image: Image.Image,
        point: tuple[float, float],
    ) -> tuple[np.ndarray, float]:
        response = self._request(
            {
                "cmd": "predict_point",
                "image_b64": _image_to_b64(image),
                "x": point[0],
                "y": point[1],
            }
        )
        return _mask_from_b64(response["mask_b64"]), float(response["score"])

    def predict_points(
        self,
        image: Image.Image,
        points: list[tuple[float, float]],
        *,
        negative_points: list[tuple[float, float]] | None = None,
    ) -> tuple[np.ndarray, float]:
        payload: dict = {
            "cmd": "predict_points",
            "image_b64": _image_to_b64(image),
            "points": [[x, y] for x, y in points],
        }
        if negative_points:
            payload["negative_points"] = [[x, y] for x, y in negative_points]
        response = self._request(payload)
        return _mask_from_b64(response["mask_b64"]), float(response["score"])


@lru_cache(maxsize=1)
def get_sam_runtime(config_path: str) -> SamRuntime:
    path = Path(config_path) if config_path else None
    return SamRuntime.load(path)
