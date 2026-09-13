"""Generate the fixed holdout OCR dataset on disk."""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from tqdm.auto import tqdm

from config import Config
from renderer.corpus import require_corpus
from renderer.paragraph_render import render_paragraph_sample
from renderer.words_gen import words_path


def holdout_manifest_path(cfg: Config) -> Path:
    return cfg.holdout.dir / "manifest.jsonl"


def generate_holdout(cfg: Config, *, force: bool = False) -> Path:
    """Render paragraph holdout samples to cfg.holdout.dir and write manifest.jsonl."""
    holdout = cfg.holdout
    manifest = holdout_manifest_path(cfg)
    if manifest.is_file() and not force:
        count = sum(1 for _ in manifest.open(encoding="utf-8"))
        print(f"holdout already exists: {manifest} ({count:,} samples)")
        print("Use --force to regenerate.")
        return manifest

    corpus = require_corpus(words_path(cfg))
    out_dir = holdout.dir
    images_dir = out_dir / "images"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    images_dir.mkdir(parents=True)

    records: list[dict[str, object]] = []
    for index in tqdm(range(holdout.num_samples), desc="Holdout", unit="sample"):
        sample_rng = random.Random(holdout.seed + index)
        sample = render_paragraph_sample(
            holdout.width,
            holdout.height,
            sample_rng,
            corpus,
            paragraph_counts=holdout.paragraph_counts,
            paragraph_len_min=holdout.paragraph_len_min,
            paragraph_len_max=holdout.paragraph_len_max,
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
