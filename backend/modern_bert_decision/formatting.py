"""Build batched encoder inputs with [SEP] segment boundaries."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerBase

from config import DecisionConfig


@dataclass(frozen=True)
class FormattedBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    option_mask: torch.Tensor


@dataclass(frozen=True)
class SequenceBoundaryIds:
    sep_id: int
    pad_id: int


def _pad_token_id(tokenizer: PreTrainedTokenizerBase) -> int:
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        sep_id = tokenizer.sep_token_id
        if sep_id is None:
            raise ValueError("tokenizer must define pad_token_id or sep_token_id")
        pad_id = sep_id
    return int(pad_id)


def resolve_sequence_boundary_ids(tokenizer: PreTrainedTokenizerBase) -> SequenceBoundaryIds:
    sep_id = tokenizer.sep_token_id
    if sep_id is None:
        raise ValueError("tokenizer must define sep_token_id")
    return SequenceBoundaryIds(sep_id=int(sep_id), pad_id=_pad_token_id(tokenizer))


def max_sequence_length(tokenizer: PreTrainedTokenizerBase, encoder_config) -> int:
    """Encoder context size (e.g. 8192 for ModernBERT-base); syncs tokenizer ``model_max_length``."""
    mpe = getattr(encoder_config, "max_position_embeddings", None)
    if mpe is None:
        model_max = getattr(tokenizer, "model_max_length", None)
        if model_max is not None and model_max < 1_000_000:
            max_len = int(model_max)
        else:
            raise ValueError(
                "encoder config must define max_position_embeddings (or tokenizer model_max_length)"
            )
    else:
        max_len = int(mpe)
    if max_len < 1:
        raise ValueError(f"invalid encoder context limit {max_len}")
    tokenizer.model_max_length = max_len
    return max_len


def _encode_field(tokenizer: PreTrainedTokenizerBase, text: str) -> list[int]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("text segment must be non-blank after stripping")
    return tokenizer.encode(
        stripped,
        add_special_tokens=False,
        truncation=False,
        max_length=None,
    )


CANDIDATE_SEGMENT_TEMPLATE = "Is the correct answer to the question is: {candidate}"
STATE_SEGMENT_PREFIX = "Context:\n"
QUESTION_SEGMENT_PREFIX = "Question:\n"


def format_state_segment(state: str) -> str:
    text = state.strip()
    if not text:
        raise ValueError("state must be non-blank after stripping")
    return f"{STATE_SEGMENT_PREFIX}{text}"


def format_question_segment(question: str) -> str:
    text = question.strip()
    if not text:
        raise ValueError("question must be non-blank after stripping")
    return f"{QUESTION_SEGMENT_PREFIX}{text}"


def format_candidate_segment(candidate: str) -> str:
    text = candidate.strip()
    if not text:
        raise ValueError("candidate must be non-blank after stripping")
    return CANDIDATE_SEGMENT_TEMPLATE.format(candidate=text)


def format_candidate_sequence_text(
    *,
    state: str,
    question: str,
    candidate: str,
) -> str:
    """Human-readable layout matching ``build_candidate_sequence`` segment order."""
    return "\n".join(
        [
            format_state_segment(state),
            f"[SEP] {format_question_segment(question)}",
            f"[SEP] {format_candidate_segment(candidate)}",
        ]
    )


def build_candidate_sequence(
    *,
    tokenizer: PreTrainedTokenizerBase,
    boundaries: SequenceBoundaryIds,
    state: str,
    question: str,
    candidate: str,
    max_len: int,
) -> list[int]:
    segments = [
        format_state_segment(state),
        format_question_segment(question),
        format_candidate_segment(candidate),
    ]
    input_ids: list[int] = []
    input_ids.extend(_encode_field(tokenizer, segments[0]))
    for text in segments[1:]:
        input_ids.append(boundaries.sep_id)
        input_ids.extend(_encode_field(tokenizer, text))
    if len(input_ids) > max_len:
        raise ValueError(
            f"sequence length {len(input_ids)} exceeds encoder context limit {max_len}; "
            "refusing to truncate"
        )
    return input_ids


def format_request(
    state: str,
    question: str,
    options: list[str],
    *,
    cfg: DecisionConfig,
    tokenizer: PreTrainedTokenizerBase,
    max_len: int,
    device: torch.device | None = None,
) -> FormattedBatch:
    return format_batch(
        [{"state": state, "question": question, "options": options}],
        cfg=cfg,
        tokenizer=tokenizer,
        max_len=max_len,
        device=device,
    )


def format_batch(
    requests: list[dict[str, object]],
    *,
    cfg: DecisionConfig,
    tokenizer: PreTrainedTokenizerBase,
    max_len: int,
    device: torch.device | None = None,
) -> FormattedBatch:
    if not requests:
        raise ValueError("requests must be non-empty")
    boundaries = resolve_sequence_boundary_ids(tokenizer)

    per_question: list[list[list[int]]] = []
    k_max = 0
    for req in requests:
        state = str(req.get("state", ""))
        question = str(req.get("question", ""))
        raw_options = req.get("options")
        if not isinstance(raw_options, list):
            raise ValueError("each request must include an options list")
        options = [str(o) for o in raw_options]
        sequences: list[list[int]] = []
        for candidate in options:
            sequences.append(
                build_candidate_sequence(
                    tokenizer=tokenizer,
                    boundaries=boundaries,
                    state=state,
                    question=question,
                    candidate=candidate,
                    max_len=max_len,
                )
            )
        per_question.append(sequences)
        k_max = max(k_max, len(sequences))

    batch_size = len(per_question)
    max_seq = max(len(seq) for seqs in per_question for seq in seqs)

    input_ids = torch.full((batch_size, k_max, max_seq), boundaries.pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, k_max, max_seq), dtype=torch.long)
    option_mask = torch.zeros((batch_size, k_max), dtype=torch.bool)

    for row, sequences in enumerate(per_question):
        for col, seq in enumerate(sequences):
            attention_mask[row, col, : len(seq)] = 1
            input_ids[row, col, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            option_mask[row, col] = True

    out = FormattedBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        option_mask=option_mask,
    )
    if device is not None:
        return FormattedBatch(
            input_ids=out.input_ids.to(device),
            attention_mask=out.attention_mask.to(device),
            option_mask=out.option_mask.to(device),
        )
    return out
