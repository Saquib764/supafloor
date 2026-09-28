"""Training loss on raw CLS logits (no sigmoid)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

GOLD_LOGIT_MIN = 0.1
WRONG_LOGIT_MAX = -0.1


def margin_logit_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Penalize gold rows with logit < 0.1 and wrong rows with logit > -0.1."""
    if logits.shape != labels.shape:
        raise ValueError(f"logits and labels shape mismatch: {tuple(logits.shape)} vs {tuple(labels.shape)}")
    labels_f = labels.to(dtype=logits.dtype)
    pos = labels_f >= 0.5
    neg = ~pos
    per = torch.zeros_like(logits)
    per[pos] = F.relu(GOLD_LOGIT_MIN - logits[pos])
    per[neg] = F.relu(logits[neg] - WRONG_LOGIT_MAX)
    return per.mean()
