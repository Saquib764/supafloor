"""Build batched encoder inputs from state, question, and option strings."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerBase

from config import DecisionConfig


@dataclass(frozen=True)
class MarkerIds:
    state: int
    question: int
    option: int


@dataclass(frozen=True)
class FormattedBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    option_positions: torch.Tensor
    option_mask: torch.Tensor


def marker_ids_from_tokenizer(tokenizer: PreTrainedTokenizerBase, cfg: DecisionConfig) -> MarkerIds:
    m = cfg.marker_tokens
    for token in (m.state, m.question, m.option):
        tid = tokenizer.convert_tokens_to_ids(token)
        if tid is None or tid == tokenizer.unk_token_id:
            raise ValueError(f"marker token not in vocabulary: {token!r}")
    return MarkerIds(
        state=tokenizer.convert_tokens_to_ids(m.state),
        question=tokenizer.convert_tokens_to_ids(m.question),
        option=tokenizer.convert_tokens_to_ids(m.option),
    )


def _require_boundary_tokens(tokenizer: PreTrainedTokenizerBase) -> tuple[int, int, int]:
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    if cls_id is None or sep_id is None:
        raise ValueError("tokenizer must define cls_token_id and sep_token_id")
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = sep_id
    return cls_id, sep_id, pad_id


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
        raise ValueError("state, question, and each option must be non-blank after stripping")
    return tokenizer.encode(stripped, add_special_tokens=False)


def build_one_sequence(
    *,
    tokenizer: PreTrainedTokenizerBase,
    markers: MarkerIds,
    cls_id: int,
    sep_id: int,
    state: str,
    question: str,
    options: list[str],
    max_len: int,
) -> tuple[list[int], list[int]]:
    if not options:
        raise ValueError("options must be a non-empty list")
    state_ids = _encode_field(tokenizer, state)
    question_ids = _encode_field(tokenizer, question)
    option_token_lists: list[list[int]] = []
    for opt in options:
        option_token_lists.append(_encode_field(tokenizer, opt))

    input_ids: list[int] = [cls_id, markers.state, *state_ids, markers.question, *question_ids]
    option_positions: list[int] = []
    for opt_ids in option_token_lists:
        option_positions.append(len(input_ids))
        input_ids.append(markers.option)
        input_ids.extend(opt_ids)
    input_ids.append(sep_id)

    if len(input_ids) > max_len:
        raise ValueError(
            f"sequence length {len(input_ids)} exceeds encoder context limit {max_len}; "
            "refusing to truncate"
        )
    return input_ids, option_positions


def format_request(
    state: str,
    question: str,
    options: list[str],
    *,
    tokenizer: PreTrainedTokenizerBase,
    markers: MarkerIds,
    max_len: int,
    device: torch.device | None = None,
) -> FormattedBatch:
    return format_batch(
        [{"state": state, "question": question, "options": options}],
        tokenizer=tokenizer,
        markers=markers,
        max_len=max_len,
        device=device,
    )


def format_batch(
    requests: list[dict[str, object]],
    *,
    tokenizer: PreTrainedTokenizerBase,
    markers: MarkerIds,
    max_len: int,
    device: torch.device | None = None,
) -> FormattedBatch:
    if not requests:
        raise ValueError("requests must be non-empty")
    cls_id, sep_id, pad_id = _require_boundary_tokens(tokenizer)

    sequences: list[list[int]] = []
    position_lists: list[list[int]] = []
    k_max = 0
    for req in requests:
        state = str(req.get("state", ""))
        question = str(req.get("question", ""))
        raw_options = req.get("options")
        if not isinstance(raw_options, list):
            raise ValueError("each request must include an options list")
        options = [str(o) for o in raw_options]
        ids, positions = build_one_sequence(
            tokenizer=tokenizer,
            markers=markers,
            cls_id=cls_id,
            sep_id=sep_id,
            state=state,
            question=question,
            options=options,
            max_len=max_len,
        )
        sequences.append(ids)
        position_lists.append(positions)
        k_max = max(k_max, len(positions))

    batch_size = len(sequences)
    max_seq = max(len(s) for s in sequences)

    input_ids = torch.full((batch_size, max_seq), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_seq), dtype=torch.long)
    option_positions = torch.zeros((batch_size, k_max), dtype=torch.long)
    option_mask = torch.zeros((batch_size, k_max), dtype=torch.bool)

    for row, (seq, positions) in enumerate(zip(sequences, position_lists, strict=True)):
        attention_mask[row, : len(seq)] = 1
        input_ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        for col, pos in enumerate(positions):
            option_positions[row, col] = pos
            option_mask[row, col] = True

    out = FormattedBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        option_positions=option_positions,
        option_mask=option_mask,
    )
    if device is not None:
        return FormattedBatch(
            input_ids=out.input_ids.to(device),
            attention_mask=out.attention_mask.to(device),
            option_positions=out.option_positions.to(device),
            option_mask=out.option_mask.to(device),
        )
    return out
