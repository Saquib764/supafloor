# Dataset manager

Single source of truth for Supafloor datasets: metadata, schema code, Hub card, upload, and download.

## Layout

```
dataset_manager/
  datasets/
    <name>/
      metadata.yaml   # hf_repo, local_root (uploadable), private, revision
      schema.py       # FEATURES + build/load/write helpers
      README.md       # Hub dataset card
  registry.py         # resolve dataset, load / upload / download
  cli.py
```

Registered today:

| Name | Role | Local folder | Hub repo |
|---|---|---|---|
| `floorplan-to-json` | Synthetic export publish/download + Florence/SAM training | `image-to-floorplan/basic_5000` | `saquiboye/floorplan-to-json` |
| `floorplan-manual` | Manual annotation publish/download + Florence training | `floorplan-manual` | `saquiboye/floorplan-manual` |
| `floorplan-manual-mask` | Manual five-class mask publish/download + FPN Five training | `floorplan-manual-mask` | `saquiboye/floorplan-manual-mask` |
| `floorplan-vae` | VAE view (`stem`, `plan_id`, `variant`, `image`, `simple_image`) | — | reads `floorplan-to-json` |
| `race` | RACE reading comprehension MCQ (`state`, `question`, `options`, `answer_index`) | `race-data` (CMU tarball) | `ehovy/race` on Hub |
| `mnli` | MNLI three-way NLI (`Premise`/`Hypothesis` state, fixed question/options) | `mnli-data` | `nyu-mll/glue` (`mnli`) |

Both uploadable datasets share the same local folder layout:

```
<input_dir>/input/<stem>.jpg
<input_dir>/output/<stem>.json
<input_dir>/output2/<stem>.jpg
```

Hub columns match for both JSON uploadable datasets (`stem`, `plan_id`, `variant`, `image`, `extraction`, `simple_image`). Manual samples always use `variant: 0` and get a random 12-char `plan_id` at upload time.

Manual mask layout (`floorplan-manual-mask`):

```
<input_dir>/input/<stem>.jpg
<input_dir>/output/<stem>/{background,room,wall,door,window}.png
```

Sample processing for trainers lives in `processing/`:

| Module | Used when `data.dataset` is |
|---|---|
| `processing/floorplan_mask.py` | `floorplan-to-json` — renders masks from extraction JSON |
| `processing/manual_mask.py` | `floorplan-manual-mask` — loads pre-rendered PNG masks |

## Upload to Hugging Face

1. Create the dataset repo on Hugging Face if it does not exist yet (`saquiboye/floorplan-to-json` or `saquiboye/floorplan-manual`).
2. Auth with a write token:

```bash
cd dataset_manager
cp .env.example .env   # set HF_TOKEN=hf_...
uv sync
```

3. Upload:

```bash
# Synthetic basic_5000 export
uv run python cli.py upload floorplan-to-json

# Manual annotations at repo root
uv run python cli.py upload floorplan-manual

# Make public (both default to private in metadata.yaml)
uv run python cli.py upload floorplan-manual --public
```

4. Download back to the local folder configured in `metadata.yaml`:

```bash
uv run python cli.py download floorplan-to-json
uv run python cli.py download floorplan-manual
```

Equivalent entry point: `uv run dataset-mgr upload floorplan-manual`.

Auth: `HF_TOKEN` in `.env` (gitignored). Load-only Hub view: `floorplan-vae` (`backend/vae_trainer`). For manual Florence/SAM training set `data.dataset: floorplan-manual` in the trainer `config.yaml`.
