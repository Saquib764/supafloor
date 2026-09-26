---
license: other
task_categories:
  - text-classification
language:
  - en
tags:
  - language-model
  - binary-classification
  - question-answering
  - gru
pretty_name: LLM Data
size_categories:
  - n<1K
---

# LLM Data

Context paragraphs with yes/no questions for binary decision training.

| Field | Type | Meaning |
|---|---|---|
| `context` | string | Background paragraph (may be long) |
| `question` | string | Yes/no question about the context |
| `output` | int | Answer: `1` = yes, `0` = no |

## Local layout

```
sentences.jsonl
```

Each line is JSON:

```json
{
  "context": "Sun revolves around earth.",
  "question": "Is this statement correct?",
  "output": 0
}
```

## GRU LLM training

Set `data.dataset: llm-data` in `backend/gru_llm/config.yaml`. The trainer loads rows via `dataset_manager.registry.load_dataset("llm-data")`.
