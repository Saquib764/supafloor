# RACE

Reading comprehension from [CMU RACE](https://www.cs.cmu.edu/~glai1/data/race/) (Lai et al.). Each row is one multiple-choice question over a passage.

## Columns

| Column | Description |
|--------|-------------|
| `state` | Passage text (`article` in the original release) |
| `question` | Question string |
| `options` | Four option strings |
| `answer_index` | Gold index 0–3 (original labels `A`–`D`) |
| `example_id` | Source id when available |

## Hub

Default Hub source: [`ehovy/race`](https://huggingface.co/datasets/ehovy/race) (`all` subset: middle + high school).

```bash
cd dataset_manager
uv run python cli.py download race

# or from the trainer package:
cd backend/modern_bert_decision
uv run modern-bert-decision download-data
```

Writes under repo-root `race-data/`:

```
race-data/train.jsonl
race-data/validation.jsonl
race-data/test.jsonl
```

Training reads these files **line-by-line** (no full in-memory Hub load).

## Local (CMU tarball)

1. Download `RACE.tar.gz` from the [CMU page](https://www.cs.cmu.edu/~glai1/data/race/).
2. Extract to repo-root `race-data/` so paths like `race-data/train/middle/*.txt` exist.
3. Set trainer `data.source: local` or `auto`.

Each `.txt` file is one JSON object with `article`, `questions`, `options`, and `answers` arrays.
