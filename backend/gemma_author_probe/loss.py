"""Cross-entropy and supervised contrastive loss on pooled embeddings."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from labels import NUM_CLASSES


def cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.dim() != 2 or int(logits.shape[-1]) != NUM_CLASSES:
        raise ValueError(f"logits must be [B, {NUM_CLASSES}], got {tuple(logits.shape)}")
    return F.cross_entropy(logits, labels.view(-1))


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """L2-normalized embeddings. Same author is positive; every other author is negative.

    Each anchor's loss is the mean log-probability of its other same-author rows
    against every other row in the batch. Returns the loss, the mean positive
    cosine, and the mean negative cosine.
    """
    if embeddings.dim() != 2:
        raise ValueError(f"embeddings must be [B, D], got {tuple(embeddings.shape)}")
    if temperature <= 0.0:
        raise ValueError("contrastive temperature must be > 0")
    authors = labels.view(-1)
    batch = int(embeddings.shape[0])
    if int(authors.shape[0]) != batch:
        raise ValueError(
            f"labels length {int(authors.shape[0])} != batch size {batch}"
        )
    if batch < 2:
        raise ValueError("contrastive loss needs at least 2 rows")

    cosine = embeddings @ embeddings.T
    same = authors[:, None] == authors[None, :]
    eye = torch.eye(batch, dtype=torch.bool, device=embeddings.device)
    positive = same & ~eye
    negative = ~same
    n_pos = positive.sum(dim=-1)
    if int(n_pos.sum()) == 0:
        raise ValueError("contrastive loss needs a repeated author in the batch")
    if not bool(negative.any().item()):
        raise ValueError("contrastive loss needs at least two authors in the batch")

    logits = cosine.masked_fill(eye, float("-inf")) / float(temperature)
    log_prob = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    per_anchor = -(log_prob.masked_fill(~positive, 0.0).sum(dim=-1) / n_pos.clamp(min=1))
    loss = per_anchor[n_pos > 0].mean()
    detached = cosine.detach()
    return (
        loss,
        detached.masked_select(positive).mean(),
        detached.masked_select(negative).mean(),
    )
