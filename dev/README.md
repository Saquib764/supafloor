# Dev scripts

Manual utilities for inspecting floor-plan generation and scoring.

Run from the repo with `uv` (service root: `backend/floorplan/`):

```bash
# 1) Generate plans as JSON + PNG into a folder
uv run --project backend/floorplan python dev/generate_floorplans.py \
  --out dev/output --count 5

# 2) Score that folder; writes README.md + per-plan *.score.md with images embedded
uv run --project backend/floorplan python dev/score_floorplans.py \
  --dir dev/output
```

Large deterministic corpus (basic score tier, no LLM) — prefer the CLI:

```bash
cd backend/floorplan
uv run floorplan dataset -n 5000 --seed 0 --score-level basic --workers 4 \
  --out ../../floorplan_dataset/basic_5000
```

| Script | Purpose |
|---|---|
| `generate_floorplans.py` | Run the generator; write `<id>.json` + `<id>.png` (+ `index.json`) |
| `score_floorplans.py` | Score each plan; write human-readable reports and `README.md` |
| `floorplan dataset` | Threshold-gated unique corpus under `floorplan_dataset/` (gitignored) |
