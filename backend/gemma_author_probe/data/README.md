Training reads **ntl_open_pool.csv** (`data.train_csv`). Evaluation uses **test.csv**; `data.test_fraction` picks that share of complete `prompt_id`s (`data.seed`).

```bash
# ntl_open_pool.csv and test.csv in data/
uv run gemma-author-probe train
```

To regenerate the CSV from parquet: `uv run gemma-author-probe export-data` (defaults to `data/ntl_open_pool.parquet`).

Parquet columns map to CSV as: `id` → `id`, `prompt_group` → `prompt_id`, `text` → `text`, `model` → `model`.

Only prompts with all 18 models are kept for training and test sampling.
