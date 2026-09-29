"""Training loss: cross-entropy against one-hot targets over valid options."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_target_cross_entropy(
    logits: torch.Tensor,
    answer_index: torch.Tensor,
    option_mask: torch.Tensor,
) -> torch.Tensor:
    """Softmax CE with a one-hot target at ``answer_index`` (masked invalid slots)."""
    if logits.dim() != 2:
        raise ValueError(f"logits must be [B, K], got {tuple(logits.shape)}")
    if answer_index.dim() != 1 or answer_index.shape[0] != logits.shape[0]:
        raise ValueError(
            f"answer_index must be [B], got {tuple(answer_index.shape)} for B={logits.shape[0]}"
        )
    if option_mask.shape != logits.shape:
        raise ValueError(f"option_mask shape {tuple(option_mask.shape)} != logits {tuple(logits.shape)}")

    masked_logits = logits.masked_fill(~option_mask, float("-inf"))
    log_probs = F.log_softmax(masked_logits.float(), dim=-1)
    gold = option_mask.gather(1, answer_index.unsqueeze(1)).squeeze(1)
    if not gold.all():
        raise ValueError("answer_index must refer to a valid (masked-in) option")
    gold_log_prob = log_probs.gather(1, answer_index.unsqueeze(1)).squeeze(1)
    return (-gold_log_prob).mean()
