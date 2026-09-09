"""Replicate image-edit cleaning pipeline for floor plans."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from config import Config
from util.canvas import load_canvas_image
from util.replicate_client import run_replicate_edit
from util.render import render_image_diff

_STEPS = (
    ("00-canvas.png", None),
    ("01-clean-structure.png", "clean"),
    ("02-openings-closed.png", "openings"),
    ("03-no-furniture.png", "furniture"),
)


def _step_path(out_dir: Path, filename: str) -> Path:
    return out_dir / filename


def _diff_path(step_path: Path) -> Path:
    return step_path.with_name(f"{step_path.stem}-diff{step_path.suffix}")


def _save_step_diff(out_dir: Path, prev_filename: str, step_path: Path) -> None:
    diff_path = _diff_path(step_path)
    if diff_path.is_file():
        return
    before = Image.open(_step_path(out_dir, prev_filename)).convert("RGB")
    after = Image.open(step_path).convert("RGB")
    diff_path.parent.mkdir(parents=True, exist_ok=True)
    render_image_diff(before, after).save(diff_path)
    print(f"wrote {diff_path}")


def run_image_edit_pipeline(
    cfg: Config,
    image_path: Path,
    *,
    skip_replicate: bool = False,
) -> tuple[Image.Image, Path]:
    if not cfg.replicate_token and not skip_replicate:
        raise ValueError("replicate token missing — set REPLICATE_API_TOKEN or replicate_token in config")

    out_dir = cfg.image_edit.output_dir / image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    canvas_path = _step_path(out_dir, "00-canvas.png")
    if canvas_path.is_file():
        image = Image.open(canvas_path).convert("RGB")
        print(f"loaded {canvas_path}")
    else:
        image = load_canvas_image(
            image_path,
            canvas_size=cfg.canvas_size,
            margin=cfg.margin,
        )
        image.save(canvas_path)
        print(f"wrote {canvas_path}")

    final_path = _step_path(out_dir, "03-no-furniture.png")
    if skip_replicate:
        if not final_path.is_file():
            raise FileNotFoundError(f"skip-replicate set but missing {final_path}")
        for step_idx, (filename, _) in enumerate(_STEPS[1:], start=1):
            _save_step_diff(out_dir, _STEPS[step_idx - 1][0], _step_path(out_dir, filename))
        print(f"loaded {final_path}")
        return Image.open(final_path).convert("RGB"), out_dir

    edit = cfg.image_edit
    for step_idx, (filename, prompt_key) in enumerate(_STEPS[1:], start=1):
        step_path = _step_path(out_dir, filename)
        prev_filename = _STEPS[step_idx - 1][0]

        if step_path.is_file():
            image = Image.open(step_path).convert("RGB")
            print(f"loaded {step_path}")
        else:
            prompt = getattr(edit.prompts, prompt_key)
            print(f"replicate ({filename}): {prompt}")
            image = run_replicate_edit(
                image,
                prompt,
                model=edit.model,
                token=cfg.replicate_token,
                quality=edit.quality,
                aspect_ratio=edit.aspect_ratio,
                output_format=edit.output_format,
                background=edit.background,
                moderation=edit.moderation,
                number_of_images=edit.number_of_images,
            )
            image.save(step_path)
            print(f"wrote {step_path}")

        _save_step_diff(out_dir, prev_filename, step_path)

    return image, out_dir
