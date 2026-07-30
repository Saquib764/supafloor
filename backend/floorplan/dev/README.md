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

# 3a) Export Prompt→Output training JSONL from a corpus folder
uv run python dev/export_training.py \
  --dir ../../floorplan_dataset/basic_5000 \
  --out ../../floorplan_dataset/basic_5000_train.jsonl

# 3b) Export Unsloth Gemma-4 E4B LoRA JSONL (chat messages + Alpaca aliases)
uv run python dev/export_unsloth_dataset.py \
  --dir ../../floorplan_dataset/basic_5000 \
  --out ../../floorplan_dataset/basic_5000_unsloth.jsonl
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
| `export_unsloth_dataset.py` | Same content wrapped for Unsloth / Gemma-4 (`messages` with role=`model`) |
| `floorplan dataset` | Threshold-gated unique corpus under `floorplan_dataset/` (gitignored) |

### Unsloth / Gemma 4 E4B LoRA format

Upload the Unsloth JSONL in Unsloth Studio (Local dataset). Each line has:

- **prompt** — natural-language room locations / adjacencies
- **schema** — room target areas + boundary `layout_polygon`
- **output** — room polygon JSON (training target)
- **messages** — ChatML for Gemma 4 IT with roles `system` / `user` / **`model`** (not `assistant`)
- **instruction** / **input** — Alpaca aliases of prompt / schema for Studio column mapping

Target model: `unsloth/gemma-4-E4B-it` (or `google/gemma-4-E4B-it`). Use chat template `gemma-4` (non-thinking for the small E4B).
