"""Build batched encoder inputs: one masked sequence per question ([SEP], [CLS] on options)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerBase

from config import (
    SCORE_TOKEN_CLS,
    SCORE_TOKEN_MASK,
    DecisionConfig,
    normalize_score_token,
)

INSTRUCTION_SEGMENT = "Label the following options as true or false to justify the question based on the context."


@dataclass(frozen=True)
class FormattedBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    score_index: torch.Tensor
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


def encoder_context_limit(tokenizer: PreTrainedTokenizerBase, encoder_config) -> int:
    mpe = getattr(encoder_config, "max_position_embeddings", None)
    if mpe is None:
        model_max = getattr(tokenizer, "model_max_length", None)
        if model_max is not None and model_max < 1_000_000:
            limit = int(model_max)
        else:
            raise ValueError(
                "encoder config must define max_position_embeddings (or tokenizer model_max_length)"
            )
    else:
        limit = int(mpe)
    if limit < 1:
        raise ValueError(f"invalid encoder context limit {limit}")
    return limit


def resolve_sequence_length(
    tokenizer: PreTrainedTokenizerBase,
    encoder_config,
    *,
    cap: int,
) -> int:
    """Apply ``model.max_len`` cap (must not exceed encoder ``max_position_embeddings``)."""
    enc_limit = encoder_context_limit(tokenizer, encoder_config)
    if cap < 1:
        raise ValueError("model.max_len must be >= 1")
    if cap > enc_limit:
        raise ValueError(f"model.max_len {cap} exceeds encoder context {enc_limit}")
    tokenizer.model_max_length = cap
    return cap


def max_sequence_length(tokenizer: PreTrainedTokenizerBase, encoder_config) -> int:
    """Encoder max context; prefer ``resolve_sequence_length`` with config cap."""
    limit = encoder_context_limit(tokenizer, encoder_config)
    tokenizer.model_max_length = limit
    return limit


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


def format_option_segment(option: str, *, mask_token: str) -> str:
    text = option.strip()
    if not text:
        raise ValueError("option must be non-blank after stripping")
    return f"{text} is {mask_token}"


def format_masked_input_text(
    *,
    state: str,
    question: str,
    options: list[str],
    mask_token: str,
    sep_token: str,
    cls_token: str,
) -> str:
    """Human-readable layout matching ``build_masked_sequence`` segment order."""
    tail: list[tuple[bool, str]] = [
        (False, format_question_segment(question)),
        (False, INSTRUCTION_SEGMENT),
        *[(True, format_option_segment(opt, mask_token=mask_token)) for opt in options],
    ]
    lines = [format_state_segment(state)]
    for cls_after_sep, text in tail:
        lines.append("")
        if cls_after_sep:
            lines.append(f"{sep_token} {cls_token} {text}")
        else:
            lines.append(f"{sep_token} {text}")
    return "\n".join(lines)


def masked_sequence_segments(
    *,
    state: str,
    question: str,
    options: list[str],
    mask_token: str,
) -> list[str]:
    segments = [
        format_state_segment(state),
        format_question_segment(question),
        INSTRUCTION_SEGMENT,
    ]
    for opt in options:
        segments.append(format_option_segment(opt, mask_token=mask_token))
    return segments


def _cls_token_id(tokenizer: PreTrainedTokenizerBase) -> int:
    cls_id = tokenizer.cls_token_id
    if cls_id is None:
        raise ValueError("tokenizer must define cls_token_id for option segments")
    return int(cls_id)


def mask_token_id(tokenizer: PreTrainedTokenizerBase) -> int:
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise ValueError("tokenizer must define mask_token_id")
    return int(mask_id)


def _append_sep_segment(
    suffix_ids: list[int],
    *,
    tokenizer: PreTrainedTokenizerBase,
    boundaries: SequenceBoundaryIds,
    text: str,
    cls_after_sep: bool = False,
) -> None:
    suffix_ids.extend(
        tokenizer.encode("\n", add_special_tokens=False, truncation=False, max_length=None)
    )
    suffix_ids.append(boundaries.sep_id)
    if cls_after_sep:
        suffix_ids.append(_cls_token_id(tokenizer))
    suffix_ids.extend(_encode_field(tokenizer, text))


def build_masked_sequence(
    *,
    tokenizer: PreTrainedTokenizerBase,
    boundaries: SequenceBoundaryIds,
    state: str,
    question: str,
    options: list[str],
    max_len: int,
) -> list[int]:
    mask_token = tokenizer.mask_token
    if not mask_token:
        raise ValueError("tokenizer must define mask_token")
    segments = masked_sequence_segments(
        state=state,
        question=question,
        options=options,
        mask_token=mask_token,
    )
    state_ids = _encode_field(tokenizer, segments[0])
    suffix_ids: list[int] = []
    for offset, text in enumerate(segments[1:], start=1):
        _append_sep_segment(
            suffix_ids,
            tokenizer=tokenizer,
            boundaries=boundaries,
            text=text,
            cls_after_sep=offset >= 3,
        )

    if len(state_ids) + len(suffix_ids) <= max_len:
        return state_ids + suffix_ids

    max_state = max_len - len(suffix_ids)
    if max_state > 0:
        state_ids = state_ids[:max_state]
        combined = state_ids + suffix_ids
        if len(combined) <= max_len:
            return combined
    return (state_ids + suffix_ids)[:max_len]


def mask_positions(input_ids: list[int], mask_id: int) -> list[int]:
    return [i for i, tid in enumerate(input_ids) if tid == mask_id]


def cls_positions(input_ids: list[int], cls_id: int) -> list[int]:
    return [i for i, tid in enumerate(input_ids) if tid == cls_id]


def _option_score_positions(
    seq: list[int],
    *,
    score_token: str,
    mask_id: int,
    cls_id: int,
    n_options: int,
) -> list[int]:
    if score_token == SCORE_TOKEN_MASK:
        positions = mask_positions(seq, mask_id)
        label = "[MASK]"
    elif score_token == SCORE_TOKEN_CLS:
        positions = cls_positions(seq, cls_id)
        label = "[CLS]"
    else:
        raise ValueError(f"unsupported score_token {score_token!r}")
    if len(positions) != n_options:
        raise ValueError(
            f"expected {n_options} {label} tokens in sequence, found {len(positions)} "
            "(context too long after truncation?)"
        )
    return positions


def format_masked_batch(
    requests: list[dict[str, object]],
    *,
    tokenizer: PreTrainedTokenizerBase,
    max_len: int,
    score_token: str = SCORE_TOKEN_MASK,
    device: torch.device | None = None,
) -> FormattedBatch:
    if not requests:
        raise ValueError("requests must be non-empty")
    score_token = normalize_score_token(score_token)
    boundaries = resolve_sequence_boundary_ids(tokenizer)
    mask_id = mask_token_id(tokenizer)
    cls_id = tokenizer.cls_token_id
    if score_token == SCORE_TOKEN_CLS and cls_id is None:
        raise ValueError("model.score_head.token is cls but tokenizer has no cls_token_id")

    sequences: list[list[int]] = []
    score_indices: list[list[int]] = []
    k_max = 0
    for req in requests:
        state = str(req.get("state", ""))
        question = str(req.get("question", ""))
        raw_options = req.get("options")
        if not isinstance(raw_options, list):
            raise ValueError("each request must include an options list")
        options = [str(o) for o in raw_options]
        seq = build_masked_sequence(
            tokenizer=tokenizer,
            boundaries=boundaries,
            state=state,
            question=question,
            options=options,
            max_len=max_len,
        )
        positions = _option_score_positions(
            seq,
            score_token=score_token,
            mask_id=mask_id,
            cls_id=int(cls_id) if cls_id is not None else -1,
            n_options=len(options),
        )
        sequences.append(seq)
        score_indices.append(positions)
        k_max = max(k_max, len(options))

    batch_size = len(sequences)
    max_seq = max(len(seq) for seq in sequences)

    input_ids = torch.full((batch_size, max_seq), boundaries.pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_seq), dtype=torch.long)
    score_index = torch.zeros((batch_size, k_max), dtype=torch.long)
    option_mask = torch.zeros((batch_size, k_max), dtype=torch.bool)

    for row, seq in enumerate(sequences):
        attention_mask[row, : len(seq)] = 1
        input_ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        for col, pos in enumerate(score_indices[row]):
            score_index[row, col] = pos
            option_mask[row, col] = True

    out = FormattedBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        score_index=score_index,
        option_mask=option_mask,
    )
    if device is not None:
        return FormattedBatch(
            input_ids=out.input_ids.to(device),
            attention_mask=out.attention_mask.to(device),
            score_index=out.score_index.to(device),
            option_mask=out.option_mask.to(device),
        )
    return out


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
    return format_masked_batch(
        requests,
        tokenizer=tokenizer,
        max_len=max_len,
        score_token=cfg.model.score_head.token,
        device=device,
    )
