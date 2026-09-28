"""Build batched encoder inputs: CLS-scored sequences with segment marker tokens."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerBase

from config import DecisionConfig


@dataclass(frozen=True)
class FormattedTrainBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


@dataclass(frozen=True)
class FormattedBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    option_mask: torch.Tensor


@dataclass(frozen=True)
class SegmentMarkerIds:
    cls_id: int
    state_id: int
    question_id: int
    options_id: int
    candidate_id: int
    pad_id: int


def _require_boundary_tokens(tokenizer: PreTrainedTokenizerBase) -> tuple[int, int]:
    cls_id = tokenizer.cls_token_id
    if cls_id is None:
        raise ValueError("tokenizer must define cls_token_id")
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        sep_id = tokenizer.sep_token_id
        if sep_id is None:
            raise ValueError("tokenizer must define pad_token_id or sep_token_id")
        pad_id = sep_id
    return cls_id, pad_id


def resolve_segment_marker_ids(tokenizer: PreTrainedTokenizerBase) -> SegmentMarkerIds:
    from model.segment_markers import marker_token_ids

    cls_id, pad_id = _require_boundary_tokens(tokenizer)
    ids = marker_token_ids(tokenizer)
    return SegmentMarkerIds(
        cls_id=cls_id,
        state_id=ids["state"],
        question_id=ids["question"],
        options_id=ids["options"],
        candidate_id=ids["candidate"],
        pad_id=pad_id,
    )


def max_sequence_length(tokenizer: PreTrainedTokenizerBase, encoder_config) -> int:
    if getattr(encoder_config, "max_position_embeddings", None) is not None:
        return int(encoder_config.max_position_embeddings)
    model_max = getattr(tokenizer, "model_max_length", None)
    if model_max is not None and model_max < 1_000_000:
        return int(model_max)
    raise ValueError("could not determine encoder context limit from config or tokenizer")


def _encode_field(tokenizer: PreTrainedTokenizerBase, text: str) -> list[int]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("text segment must be non-blank after stripping")
    return tokenizer.encode(stripped, add_special_tokens=False)


def format_all_options(options: list[str]) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lines: list[str] = []
    for index, opt in enumerate(options):
        label = letters[index] if index < len(letters) else str(index + 1)
        lines.append(f"({label}) {opt}")
    return "\n".join(lines)


def format_candidate_sequence_text(
    *,
    state: str,
    question: str,
    options: list[str],
    candidate: str,
) -> str:
    """Human-readable layout matching ``build_candidate_sequence`` segment order."""
    return "\n".join(
        [
            "[CLS]",
            f"[state] {state.strip()}",
            f"[question] {question.strip()}",
            f"[options]\n{format_all_options(options)}",
            f"[candidate] {candidate.strip()}",
        ]
    )


def build_candidate_sequence(
    *,
    tokenizer: PreTrainedTokenizerBase,
    markers: SegmentMarkerIds,
    state: str,
    question: str,
    options: list[str],
    candidate: str,
    max_len: int,
) -> list[int]:
    if not options:
        raise ValueError("options must be a non-empty list")
    segments: list[tuple[int, str]] = [
        (markers.state_id, state),
        (markers.question_id, question),
        (markers.options_id, format_all_options(options)),
        (markers.candidate_id, candidate),
    ]
    input_ids: list[int] = [markers.cls_id]
    for marker_id, text in segments:
        input_ids.append(marker_id)
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


def format_train_batch(
    rows: list[tuple[str, str, list[str], str]],
    *,
    cfg: DecisionConfig,
    tokenizer: PreTrainedTokenizerBase,
    max_len: int,
    device: torch.device | None = None,
) -> FormattedTrainBatch:
    if not rows:
        raise ValueError("rows must be non-empty")
    markers = resolve_segment_marker_ids(tokenizer)
    sequences: list[list[int]] = []
    for state, question, options, candidate in rows:
        sequences.append(
            build_candidate_sequence(
                tokenizer=tokenizer,
                markers=markers,
                state=state,
                question=question,
                options=options,
                candidate=candidate,
                max_len=max_len,
            )
        )
    batch_size = len(sequences)
    max_seq = max(len(seq) for seq in sequences)
    input_ids = torch.full((batch_size, max_seq), markers.pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_seq), dtype=torch.long)
    for row, seq in enumerate(sequences):
        attention_mask[row, : len(seq)] = 1
        input_ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    out = FormattedTrainBatch(input_ids=input_ids, attention_mask=attention_mask)
    if device is not None:
        return FormattedTrainBatch(
            input_ids=out.input_ids.to(device),
            attention_mask=out.attention_mask.to(device),
        )
    return out


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
    markers = resolve_segment_marker_ids(tokenizer)

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
                    markers=markers,
                    state=state,
                    question=question,
                    options=options,
                    candidate=candidate,
                    max_len=max_len,
                )
            )
        per_question.append(sequences)
        k_max = max(k_max, len(sequences))

    batch_size = len(per_question)
    max_seq = max(len(seq) for seqs in per_question for seq in seqs)

    input_ids = torch.full((batch_size, k_max, max_seq), markers.pad_id, dtype=torch.long)
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


def required_marker_tokens() -> tuple[str, ...]:
    from model.segment_markers import MARKER_TOKEN_STRINGS

    return MARKER_TOKEN_STRINGS
