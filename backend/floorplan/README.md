# FloorPlan Engine Service

FastAPI backend for the Supafloor floor plan viewer and dataset engine.

## Quick start

Requires [uv](https://docs.astral.sh/uv/).

```bash
cd backend/floorplan
uv sync --group dev
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Alternative via CLI:

```bash
uv run floorplan serve --reload --port 8000
```

## Tests

```bash
uv run pytest -q
```

## Dev scripts

```bash
uv run python dev/generate_floorplans.py --out dev/output/plans --count 5
uv run python dev/score_floorplans.py --dir dev/output/plans
uv run python dev/export_training.py \
  --dir ../../floorplan_dataset/basic_5000 \
  --out ../../floorplan_dataset/basic_5000_train.jsonl
```

See `dev/README.md`.

## Layout

```
backend/floorplan/          # service root (this directory)
  app/                      # FastAPI server shell
  dev/                      # manual generate / score / export scripts
  floorplan/                # floor plan purpose module
    api/                    # /floorplan/* routes
    dataset/                # sample generator
    geometry/ topology/ …
```
