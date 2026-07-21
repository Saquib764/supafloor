---
name: sde
description: >-
  Core SDE coding standards for the Supafloor / FloorPlan Dataset Engine repo.
  Always apply on every coding task in this repository: keep logic simple, write
  only what the user asked for, prefer efficient solutions, and delete dead or
  no-op code. Use when writing, editing, reviewing, or refactoring any backend,
  frontend, or dev-script code.
---

# SDE Skill — Supafloor FloorPlan Engine

**Always use this skill** for work in this repository.

## Principles (non-negotiable)

1. **Do not overcomplicate** code or logic. Prefer the smallest clear design that works.
2. **Write simple code** that achieves the user’s request — nothing more.
3. **Write efficient code** — avoid unnecessary allocations, nested scans, and redundant work when a direct approach exists.
4. **Always avoid dead and no-op code** — no unused imports, unused variables, empty stubs, commented-out blocks, or pass-through wrappers that add no behavior.

### Practical checks before finishing

- Does every new function / branch serve the stated request?
- Can any abstraction, helper, or config be removed without losing behavior?
- Are there leftover prints, TODOs that do nothing, or duplicated paths?
- Prefer one straightforward path over speculative generality.

## Repository folder structure

Encode and respect this layout. Put new code in the matching folder; do not invent parallel trees.

```
/
├── README.md
├── .gitignore
├── .cursor/
│   └── skills/
│       └── sde/
│           └── SKILL.md          # this skill
│
├── backend/                      # Python package root (uv)
│   ├── pyproject.toml
│   └── floorplan/                # FloorPlan Dataset Engine
│       ├── api/                  # FastAPI app + routes
│       ├── assets/               # static engine assets
│       ├── dataset/              # sample / corpus generator
│       ├── evaluator/            # rules, furniture, scoring
│       ├── geometry/             # Shapely ops + authoring helpers
│       ├── importer/             # BW image → FloorPlan
│       ├── mutations/            # mutation ops + pipeline
│       ├── renderer/             # SVG / PNG rendering
│       ├── schemas/              # enums + immutable domain models
│       ├── services/             # engine facade, boundary adapt
│       ├── templates/            # apartment templates + instantiators
│       ├── tests/                # pytest suite
│       ├── topology/             # adjacency, reachability, hashes
│       ├── cli.py                # CLI entry (uv run floorplan …)
│       └── __init__.py
│
├── frontend/                     # Viewer / debug UI
│   ├── package.json
│   ├── public/
│   │   └── floorplan-viewer.html # standalone Vue viewer
│   └── app/
│       ├── types/
│       │   └── floorplan.ts
│       ├── pages/
│       │   └── floorplan/
│       │       └── index.vue
│       ├── components/
│       │   └── floorplan/        # Canvas, Inspector, Scores
│       └── composables/
│           └── floorplan/        # API client
│
└── dev/                          # Manual check scripts + outputs
    ├── README.md
    ├── generate_floorplans.py    # write JSON + PNG to a folder
    ├── score_floorplans.py       # score folder → README reports
    └── output/                   # tracked sample outputs
```

## Domain placement rules

| Change type | Put it in |
|---|---|
| Models / enums | `backend/floorplan/schemas/` |
| Polygon / line helpers | `backend/floorplan/geometry/` |
| Adjacency / hashes | `backend/floorplan/topology/` |
| Templates / instantiate | `backend/floorplan/templates/` |
| Mutate ops | `backend/floorplan/mutations/` |
| Rules / scores | `backend/floorplan/evaluator/` |
| SVG/PNG | `backend/floorplan/renderer/` |
| HTTP API | `backend/floorplan/api/` |
| Orchestration | `backend/floorplan/services/` |
| Dataset export | `backend/floorplan/dataset/` |
| Manual scripts | `dev/` |
| Viewer UI | `frontend/app/.../floorplan/` |

## Geometry contract (keep simple)

- Floor plan = separate lists: **rooms**, **doors**, **windows**
- Room = `name` + **closed valid polygon**
- Door / window = **line** (`start`, `end`)
- Prefer `make_room` / `make_door_line` / `make_window_line` over ad-hoc construction

## Tooling

- Backend: **uv** (`uv sync --group dev`, `uv run …`)
- Tests: `uv run pytest -q` from `backend/`
- Dev checks: `dev/generate_floorplans.py`, `dev/score_floorplans.py`
