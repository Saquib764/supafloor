Place **train.csv** here (from [Name That LLM](https://www.kaggle.com/competitions/name-that-llm/data)).

Training keeps complete prompts only (one row per model). `data.test_fraction: 0` fits every complete prompt. A fraction above zero holds out that share of `prompt_id`s (`data.seed`). Competition **test.csv** is scored in `kaggle_submission.ipynb`.

```bash
cd ../modern_bert_decision && uv run modern-bert-decision download-data
```

Or set `data.root: data` in `config.yaml` and place **train.csv** here.
