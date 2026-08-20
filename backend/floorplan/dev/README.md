# Dev scripts

Manual utilities for inspecting floor-plan generation and scoring.

Run from the service root (`backend/floorplan/`):

```bash
cd backend/floorplan

# 1) Generate plans as JSON + PNG into a folder
uv run python dev/generate_floorplans.py \
  --out dev/output/plans --count 5

# 2) Score that folder; writes README.md + per-plan *.score.md with images embedded
uv run python dev/score_floorplans.py \
  --dir dev/output/plans

# 3) Export Prompt→Output training JSONL from a corpus folder
uv run python dev/export_training.py \
  --dir ../../floorplan_dataset/basic_5000 \
  --out ../../floorplan_dataset/basic_5000_train.jsonl

# 4) Image → rooms (vision). Writes JSON + debug PNGs
uv run python dev/extract_floorplan.py --image out/fp.webp --out out

# 5) FloorPlan JSON → realistic PNG (uses extraction transform)
uv run python dev/export_realistic.py \
  --input dev/output/plans/9044f6156937/floorplan.json

# 6) Corpus → image-to-floorplan training pairs
uv run python dev/export_image_to_floorplan.py \
  --dir ../../floorplan_dataset/basic_5000 \
  --out ../../image-to-floorplan/basic_5000

# 7) Aggregate transform stats YAML → Markdown summary
uv run python dev/summarize_dataset_stats.py
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
| `export_training.py` | Corpus JSON → training JSONL (`input.prompt` + `input.schema` → `output` rooms) |
| `extract_floorplan.py` | Floor-plan image → room polygons (vision LLM) |
| `export_realistic.py` | FloorPlan JSON → realistic 1024 PNG |
| `export_image_to_floorplan.py` | Corpus JSON → input PNG + extraction JSON training pairs |
| `summarize_dataset_stats.py` | `stats/*.yml` → Markdown summary (`STATS.md`) |
| `floorplan dataset` | Threshold-gated unique corpus under `floorplan_dataset/` (gitignored) |
