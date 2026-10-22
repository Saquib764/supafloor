"""Ground-truth model names (Name That LLM Kaggle competition)."""

from __future__ import annotations

MODEL_NAMES: tuple[str, ...] = (
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-6-astra",
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "claude-fable-5",
    "claude-opus-4-6",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    "qwen3.8-max",
    "glm-5.3",
    "glm-5.3-flash",
    "kimi-k3",
)

NUM_CLASSES = 18
if len(MODEL_NAMES) != NUM_CLASSES:
    raise RuntimeError(f"expected {NUM_CLASSES} model names, got {len(MODEL_NAMES)}")
_LABEL_TO_INDEX = {name: i for i, name in enumerate(MODEL_NAMES)}


def model_to_index(name: str) -> int:
    key = str(name).strip()
    if key not in _LABEL_TO_INDEX:
        raise ValueError(f"unknown model {name!r}")
    return _LABEL_TO_INDEX[key]


def index_to_model(index: int) -> str:
    i = int(index)
    if not 0 <= i < NUM_CLASSES:
        raise ValueError(f"label index {index} out of range")
    return MODEL_NAMES[i]


def model_family(model_name: str) -> str:
    name = str(model_name).strip().lower()
    if name.startswith("gpt-"):
        return "gpt"
    if name.startswith("gemini-"):
        return "gemini"
    if name.startswith("claude-"):
        return "claude"
    if name.startswith("qwen"):
        return "qwen"
    if name.startswith("glm-"):
        return "glm"
    if name.startswith("kimi-"):
        return "kimi"
    raise ValueError(f"unknown model family for {model_name!r}")


def family_for_label_index(index: int) -> str:
    return model_family(index_to_model(index))


MODEL_FAMILIES: tuple[str, ...] = tuple(dict.fromkeys(model_family(n) for n in MODEL_NAMES))


def balanced_accuracy(gold_label_indices: list[int], pred_label_indices: list[int]) -> float:
    """Mean per-class recall over all candidate models."""
    if len(gold_label_indices) != len(pred_label_indices):
        raise ValueError("gold and pred label lists must have the same length")
    recalls: list[float] = []
    for i in range(NUM_CLASSES):
        gold_n = sum(1 for g in gold_label_indices if int(g) == i)
        if gold_n == 0:
            recalls.append(0.0)
            continue
        ok = sum(
            1
            for g, p in zip(gold_label_indices, pred_label_indices, strict=True)
            if int(g) == i and int(p) == i
        )
        recalls.append(ok / gold_n)
    return sum(recalls) / NUM_CLASSES
