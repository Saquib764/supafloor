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

## Rules of thumb (encode these)

### Design

- **One source of truth.** Don’t re-validate what creation already guarantees (e.g. room polygons are closed on construct — don’t rebuild/recheck every score unless necessary).
- **Name things for what they do.** If `move_wall` only translates a room, don’t pretend it’s a wall editor. Prefer honest names over legacy fiction.
- **No fake diversity.** Don’t ship template aliases that only rename geometry. Either implement the layout or don’t list it as distinct.
- **Don’t thread unused fields.** If authors don’t create `walls` / `zones`, don’t pass them through every mutator “just in case.”
- **Aliases are debt.** Prefer one function (`topology_hash`) over duplicate metadata keys that mean the same thing.

### Efficiency

- **Build polygons once per call.** Never call `touches` and `shared_edge` back-to-back on the same pair (each rebuilds Shapely geometry).
- **Don’t double-run the rule engine.** If you need validity and a score, call `score_plan` once and derive hard failures from its rules.
- **Reuse linear math when affine is uniform.** Bbox stretch is a point formula — don’t wrap every room vertex through Shapely affinity.
- **O(n²) room pairs are OK for small apartments; don’t multiply work inside the pair loop.**
- **Avoid allocating wrappers for trivial math.** Door/window length/midpoint from `start`/`end` directly — no temporary `LineSegment` unless needed.

### Cleanup hygiene

- Delete unused params (`samples` that are never read).
- Delete unused exports (`fit_polygon_in_boundary` with no callers).
- Collapse near-duplicate functions (resize / expand living / shrink hallway → one scale helper).
- Prefer `model_copy(update=...)` over dump→rebuild when only a few fields change.

### API / product

- Keep the geometry contract boring and strict:
  - Floor plan = `rooms` + `doors` + `windows`
  - Room = `name` + closed valid polygon
  - Door / window = line (`start`, `end`)
- Hard rules reject; soft rules explain. Every rule returns score + reason (+ repair when useful).
- Dev scripts stay thin: generate files, score files — no business logic hidden in `dev/`.

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
├── dataset_manager/              # Dataset registry (schema + metadata + Hub I/O)
│   ├── datasets/<name>/          # metadata.yaml, schema.py, README.md
│   ├── registry.py
│   └── cli.py                    # uv run python cli.py upload|download <name>
│
├── backend/
│   ├── floorplan/                # Python service root (uv)
│       ├── pyproject.toml
│       ├── README.md
│       ├── cli.py                # CLI entry (uv run floorplan …)
│       ├── config.yaml
│       ├── app/                  # FastAPI server shell + /floorplan routes
│       │   ├── main.py
│       │   ├── routes.py
│       │   ├── config.py         # env / CORS / API keys
│       │   └── tests/
│       ├── config/               # engine settings from config.yaml
│       ├── schemas/              # enums, domain models, units, constants
│       ├── geometry/             # Shapely ops + authoring helpers
│       ├── topology/             # adjacency, reachability, hashes
│       ├── requirements/         # boundary-first plan generation
│       │   ├── llm/              # generation LLM (centers, graph, areas)
│       │   └── prompts/          # generation prompt YAMLs
│       ├── evaluator/            # rules, scoring
│       ├── renderer/             # schematic SVG / PNG
│       ├── dataset/              # corpus generator + plan store
│       ├── extraction/           # image ↔ rooms (inference + training transform)
│       ├── export/               # realistic PNG
│       └── dev/                  # manual scripts + outputs
│           ├── README.md
│           ├── generate_floorplans.py
│           ├── score_floorplans.py
│           ├── export_training.py
│           └── output/
│   └── florence_trainer/         # Florence-2 LoRA (image → room polygons)
│       ├── pyproject.toml
│       ├── README.md
│       ├── config.yaml
│       ├── cli.py                # uv run florence-train train|infer
│       ├── loc.py                # infer: loc-token decode
│       ├── data.py
│       ├── model.py
│       ├── train.py
│       └── infer.py
│
└── frontend/                     # Viewer / debug UI
    ├── package.json
    ├── public/
    │   └── floorplan-viewer.html
    └── app/
        ├── types/floorplan.ts
        ├── pages/                # index, extract, dataset
        ├── components/
        └── composables/
```

## Domain placement rules

| Change type | Put it in |
|---|---|
| Models / enums / units | `backend/floorplan/schemas/` |
| Polygon / line helpers | `backend/floorplan/geometry/` |
| Adjacency / hashes | `backend/floorplan/topology/` |
| Growing a plan from a boundary | `backend/floorplan/requirements/` |
| Generation LLM / prompts | `backend/floorplan/requirements/llm/` and `requirements/prompts/` |
| Rules / scores | `backend/floorplan/evaluator/` |
| Schematic SVG/PNG | `backend/floorplan/renderer/` |
| Realistic PNG | `backend/floorplan/export/` |
| Image ↔ rooms / extraction JSON | `backend/floorplan/extraction/` |
| Dataset corpus | `backend/floorplan/dataset/` |
| Hub dataset registry / upload / download | `dataset_manager/` |
| Engine config.yaml | `backend/floorplan/config/` |
| HTTP API + server shell | `backend/floorplan/app/` |
| CLI | `backend/floorplan/cli.py` |
| Manual scripts | `backend/floorplan/dev/` |
| Florence-2 LoRA (image → rooms) | `backend/florence_trainer/` |
| Viewer UI | `frontend/app/` |

## Tooling

- Backend: **uv** from `backend/floorplan/` (`uv sync --group dev`, `uv run uvicorn app.main:app --reload`)
- Florence trainer: **uv** from `backend/florence_trainer/` (`uv sync`, `uv run florence-train train`)
- Tests: `uv run pytest -q` from `backend/floorplan/`
- Dev checks: `dev/generate_floorplans.py`, `dev/score_floorplans.py`, `dev/export_training.py` (from `backend/floorplan/`)
