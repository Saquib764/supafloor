# Dev scripts

Manual utilities for inspecting floor-plan generation and scoring.

Run from the repo with `uv` (backend package):

```bash
# 1) Generate plans as JSON + PNG into a folder
uv run --directory backend python ../dev/generate_floorplans.py \
  --out ../dev/output --n 5

# 2) Score that folder; writes README.md + per-plan *.score.md with images embedded
uv run --directory backend python ../dev/score_floorplans.py \
  --dir ../dev/output
```

| Script | Purpose |
|---|---|
| `generate_floorplans.py` | Run the generator; write `<id>.json` + `<id>.png` (+ `index.json`) |
| `score_floorplans.py` | Score each plan; write human-readable reports and `README.md` |
