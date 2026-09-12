"""Generate a fixed holdout OCR dataset on disk."""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from tqdm.auto import tqdm

from config import Config
from data import render_text_sample


def manifest_path(cfg: Config) -> Path:
    return cfg.holdout.dir / "manifest.jsonl"


def generate_holdout(cfg: Config, *, force: bool = False) -> Path:
    """Render holdout samples to cfg.holdout.dir and write manifest.jsonl."""
    out_dir = cfg.holdout.dir
    manifest = manifest_path(cfg)
    if manifest.is_file() and not force:
        count = sum(1 for _ in manifest.open(encoding="utf-8"))
        print(f"holdout already exists: {manifest} ({count:,} samples)")
        print("Use --force to regenerate.")
        return manifest

    images_dir = out_dir / "images"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    images_dir.mkdir(parents=True)

    holdout = cfg.holdout
    records: list[dict[str, object]] = []
    for index in tqdm(range(holdout.num_samples), desc="Holdout", unit="sample"):
        sample_rng = random.Random(holdout.seed + index)
        sample = render_text_sample(
            holdout.width,
            holdout.height,
            sample_rng,
            line_counts=holdout.line_counts,
            sentence_len_min=holdout.sentence_len_min,
            sentence_len_max=holdout.sentence_len_max,
            variations=holdout.variations,
        )
        image_name = f"{index:06d}.png"
        sample.image.save(images_dir / image_name, format="PNG")
        records.append(
            {
                "id": index,
                "image": f"images/{image_name}",
                "text": sample.text,
                "width": holdout.width,
                "height": holdout.height,
            }
        )

    with manifest.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"holdout → {manifest} ({len(records):,} samples)")
    return manifest
