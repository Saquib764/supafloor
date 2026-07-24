# Dev scripts

Manual utilities for inspecting floor-plan generation and scoring.

Run from the repo with `uv` (service root: `backend/floorplan/`):

```bash
# 1) Generate plans as JSON + PNG into a folder
uv run --project backend/floorplan python dev/generate_floorplans.py \
  --out dev/output/plans --count 5

# 2) Score that folder; writes README.md + per-plan *.score.md with images embedded
uv run --project backend/floorplan python dev/score_floorplans.py \
  --dir dev/output

# 3) Export prompt / schema / output JSONL for Unsloth Gemma-4 E4B LoRA
uv run --project backend/floorplan python dev/export_unsloth_dataset.py \
  --dir dev/output/plans \
  --out dev/output/unsloth/gemma4_e4b_sft.jsonl
```

Large deterministic corpus (basic score tier, no LLM) — prefer the CLI:

```bash
cd backend/floorplan
uv run floorplan dataset -n 5000 --seed 0 --score-level basic --workers 4 \
  --out ../../floorplan_dataset/basic_5000

# Then export for Unsloth:
uv run --project backend/floorplan python ../../dev/export_unsloth_dataset.py \
  --dir ../../floorplan_dataset/basic_5000 \
  --out ../../floorplan_dataset/basic_5000_unsloth.jsonl
```

| Script | Purpose |
|---|---|
| `generate_floorplans.py` | Run the generator; write `<id>.json` + `<id>.png` (+ `index.json`) |
| `score_floorplans.py` | Score each plan; write human-readable reports and `README.md` |
| `export_unsloth_dataset.py` | Plan JSON → Unsloth JSONL (prompt/schema/output + Gemma-4 `messages`) |
| `floorplan dataset` | Threshold-gated unique corpus under `floorplan_dataset/` (gitignored) |

### Unsloth / Gemma 4 E4B LoRA format

Upload the JSONL in Unsloth Studio (Local dataset). Each line has:

- **prompt** — natural-language room locations / adjacencies
- **schema** — room target areas + boundary `layout_polygon`
- **output** — room polygon JSON (training target)
- **messages** — ChatML for Gemma 4 IT with roles `system` / `user` / **`model`** (not `assistant`)
- **instruction** / **input** — Alpaca aliases of prompt / schema for Studio column mapping

Target model: `unsloth/gemma-4-E4B-it` (or `google/gemma-4-E4B-it`). Use chat template `gemma-4` (non-thinking for the small E4B).
