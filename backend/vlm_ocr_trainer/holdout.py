"""Generate the fixed holdout OCR dataset on disk."""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from tqdm.auto import tqdm

from config import Config
from data import sample_random_text
from renderer.paragraph_render import render_paragraph_document
from vocab import TRAINING_CHARSET


def holdout_manifest_path(cfg: Config) -> Path:
    return cfg.holdout.dir / "manifest.jsonl"


def generate_holdout(cfg: Config, *, force: bool = False) -> Path:
    """Render holdout samples to cfg.holdout.dir and write manifest.jsonl."""
    holdout = cfg.holdout
    manifest = holdout_manifest_path(cfg)
    if manifest.is_file() and not force:
        count = sum(1 for _ in manifest.open(encoding="utf-8"))
        print(f"holdout already exists: {manifest} ({count:,} samples)")
        print("Use --force to regenerate.")
        return manifest

    charset = TRAINING_CHARSET
    out_dir = holdout.dir
    images_dir = out_dir / "images"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    images_dir.mkdir(parents=True)

    records: list[dict[str, object]] = []
    for index in tqdm(range(holdout.num_samples), desc="Holdout", unit="sample"):
        sample_rng = random.Random(holdout.seed + index)
        text = sample_random_text(
            sample_rng,
            charset,
            max_text_chars=holdout.max_text_len,
        )
        sample = render_paragraph_document(
            holdout.width,
            holdout.height,
            sample_rng,
            [text],
            cfg.data.variations.final,
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
