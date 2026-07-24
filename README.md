# Supafloor — FloorPlan Dataset Engine V1

Deterministic residential floor plan engine for generating, validating, rendering, and scoring floor plans **without an LLM**.

> Code guarantees correctness. Human-designed rules capture architectural knowledge. AI learns later from deterministic examples.

## Layout

```
backend/floorplan/              # Python service root (uv project, like mission-pluto/thefluxtrain)
  pyproject.toml
  app/                          # FastAPI server shell (main.py, /health)
  floorplan/                    # FloorPlan purpose module
    api/                        # /floorplan/* routes
    schemas/                    # domain models + API request bodies
    geometry/                   # Shapely-backed immutable geometry ops
    topology/                   # Adjacency, reachability, diversity hashes
    evaluator/                  # Hard/soft rules + weighted scoring
    renderer/                   # SVG / PNG / layer overlays
    importer/                   # BW image → FloorPlan (OpenCV)
    dataset/                    # End-to-end sample & corpus generator
    services/                   # Engine facade
    requirements/               # Boundary-first plan generation

frontend/app/
  pages/                     Nuxt viewer + importer pages
  components/                Canvas, list, scores
  composables/               API client
  types/floorplan.ts
frontend/public/floorplan-viewer.html   Standalone Vue viewer
```

## Quick start

Requires [uv](https://docs.astral.sh/uv/).

```bash
cd backend/floorplan
uv sync --group dev

# API (same as mission-pluto: uvicorn on app.main)
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Tests
uv run pytest -q

# Generate a basic-level deterministic corpus (no LLM)
uv run floorplan dataset -n 5000 --seed 0 --score-level basic --workers 4 \
  --out ../../floorplan_dataset/basic_5000
```

Open the viewer: `frontend/public/floorplan-viewer.html` (expects API at `http://localhost:8000`).

## Dev scripts

```bash
# Generate JSON + PNG into a folder
uv run --project backend/floorplan python dev/generate_floorplans.py --out dev/output --count 5

# Score that folder → README.md with embedded images + human-readable reports
uv run --project backend/floorplan python dev/score_floorplans.py --dir dev/output
```

## V1 capabilities

| Capability | Status |
|---|---|
| Geometry library (Shapely) | Done |
| SVG / PNG renderer + layers | Done |
| Interactive viewer | Done |
| BW image importer | Done |
| Hard/soft rule validator | Done |
| Weighted scoring | Done |
| Dataset generator + diversity hashes | Done |

## Philosophy

1. Geometry is always validated by code. Creators always emit **closed, valid room polygons**.
2. A floor plan is primarily three separate lists: **rooms** (named polygons), **doors** (lines), and **windows** (lines).
3. Rules return **score + reason + severity + repair suggestion** (never only pass/fail).
4. Dataset samples store boundary, requirements, topology, plan, scores, SVG/PNG, and metadata.

## API highlights

- `GET  /health` — service health
- `POST /floorplan/render` → `{ svg, png, metadata }`
- `POST /floorplan/evaluate`
- `POST /floorplan/import`
- `POST /floorplan/generate`
- `GET  /floorplan/plans` — list saved plans
- `POST /floorplan/plans/{id}/load` — load plan + scores for viewer
- `DELETE /floorplan/plans/{id}`
