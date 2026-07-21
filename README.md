# Supafloor — FloorPlan Dataset Engine V1

Deterministic residential floor plan engine for generating, mutating, validating, rendering, and scoring floor plans **without an LLM**.

> Code guarantees correctness. Human-designed rules capture architectural knowledge. AI learns later from deterministic examples.

## Layout

```
backend/floorplan/
  api/          FastAPI routes (POST /floorplan/render, …)
  geometry/     Shapely-backed immutable geometry ops
  topology/     Adjacency, reachability, diversity hashes
  templates/    Studio → 4BR / duplex logical templates
  mutations/    Mirror, resize, swap, insert, …
  evaluator/    Hard/soft rules + weighted scoring + furniture
  renderer/     SVG / PNG / layer overlays
  importer/     BW image → FloorPlan (OpenCV)
  dataset/      End-to-end sample & corpus generator
  services/     Boundary adaptation + engine facade
  tests/

frontend/app/
  pages/floorplan/           Nuxt-style viewer page
  components/floorplan/      Canvas, inspector, scores
  composables/floorplan/     API client
  types/floorplan.ts
frontend/public/floorplan-viewer.html   Standalone Vue viewer
```

## Quick start

Requires [uv](https://docs.astral.sh/uv/).

```bash
cd backend
uv sync --group dev

# API
uv run floorplan serve --port 8000

# Tests
uv run pytest -q

# Generate a small dataset
uv run floorplan dataset -n 20 --out ../floorplan_dataset
```

Open the viewer: `frontend/public/floorplan-viewer.html` (expects API at `http://localhost:8000`).

## V1 capabilities

| Capability | Status |
|---|---|
| Geometry library (Shapely) | Done |
| SVG / PNG renderer + layers | Done |
| Interactive viewer | Done |
| BW image importer | Done |
| Template library | Done |
| Boundary adaptation | Done |
| Mutation engine | Done |
| Hard/soft rule validator | Done |
| Furniture simulator | Done |
| Weighted scoring | Done |
| Dataset generator + diversity hashes | Done |

## Philosophy

1. Geometry is always validated by code.
2. Rules return **score + reason + severity + repair suggestion** (never only pass/fail).
3. Dataset samples store boundary, requirements, topology, plan, scores, mutation history, SVG/PNG, and metadata.

## API highlights

- `GET  /floorplan/templates`
- `POST /floorplan/instantiate`
- `POST /floorplan/render` → `{ svg, png, metadata }`
- `POST /floorplan/evaluate`
- `POST /floorplan/mutate`
- `POST /floorplan/import`
- `POST /floorplan/generate`
- `POST /floorplan/dataset`
- `POST /floorplan/demo`
