"""Export NTL open pool parquet to competition-style train CSV."""

from __future__ import annotations

import csv
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm

from labels import model_to_index

CSV_COLUMNS = ("id", "prompt_id", "text", "model")
PARQUET_COLUMNS = ("id", "text", "model", "prompt_group")
DEFAULT_PARQUET_NAME = "ntl_open_pool.parquet"


def export_ntl_open_pool(parquet_path: Path, csv_path: Path) -> int:
    """Map parquet rows to id,prompt_id,text,model and write UTF-8 CSV."""
    if not parquet_path.is_file():
        raise FileNotFoundError(f"missing {parquet_path}")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(parquet_path, columns=list(PARQUET_COLUMNS))
    n = table.num_rows
    if n == 0:
        raise ValueError(f"no rows in {parquet_path}")

    ids = table.column("id")
    texts = table.column("text")
    models = table.column("model")
    prompt_groups = table.column("prompt_group")

    written = 0
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for i in tqdm(range(n), desc="export ntl open pool", unit="row"):
            model = str(models[i].as_py()).strip()
            model_to_index(model)
            text = str(texts[i].as_py() or "").strip()
            if not text:
                continue
            writer.writerow(
                {
                    "id": str(ids[i].as_py()).strip(),
                    "prompt_id": str(prompt_groups[i].as_py()).strip(),
                    "text": text,
                    "model": model,
                }
            )
            written += 1

    if written == 0:
        raise ValueError(f"no rows written to {csv_path}")
    return written


def main(parquet_path: Path, csv_path: Path) -> None:
    n = export_ntl_open_pool(parquet_path, csv_path)
    print(f"wrote {n} rows to {csv_path}", flush=True)


if __name__ == "__main__":
    import argparse

    from config import CONFIG_PATH, load_config

    parser = argparse.ArgumentParser(description="Export ntl_open_pool.parquet to train CSV")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--parquet", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    parquet = args.parquet or (cfg.data.root / DEFAULT_PARQUET_NAME)
    csv_out = args.csv or (cfg.data.root / cfg.data.train_csv)
    main(parquet, csv_out)
