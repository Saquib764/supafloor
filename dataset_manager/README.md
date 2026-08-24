# Dataset manager

Single source of truth for Supafloor datasets: metadata, schema code, Hub card, upload, and download.

## Layout

```
dataset_manager/
  datasets/
    <name>/
      metadata.yaml   # hf_repo, local_root (uploadable), private, revision
      schema.py       # FEATURES + build/load/write helpers
      README.md       # Hub dataset card
  registry.py         # resolve dataset, load / upload / download
  cli.py
```

Registered today:

| Name | Role |
|---|---|
| `floorplan-to-json` | Full Hub publish/download (`stem`, `plan_id`, `variant`, `image`, `extraction`) |
| `florence-rooms` | Training view (`stem`, `plan_id`, `variant`, `image`, `extraction`) |

## Usage

```bash
cd dataset_manager
uv sync

uv run python cli.py upload floorplan-to-json
uv run python cli.py download floorplan-to-json
# or: uv run dataset-mgr upload floorplan-to-json
```

Auth: `HF_TOKEN` in `.env` (gitignored). Load-only views such as `florence-rooms` are used by `backend/florence_trainer` (`data.dataset: florence-rooms`).
