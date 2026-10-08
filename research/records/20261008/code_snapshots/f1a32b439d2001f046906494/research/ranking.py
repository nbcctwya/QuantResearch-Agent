"""Daily ranking objectives for the unchanged Top30 backtest."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def topk_membership_loss(scores, labels, topk=30):
    """Compare true training top-k stocks with the remaining stocks of one date.

    All eligible pairs are used deterministically. Ties at the selection
    boundary share fractional membership; equal-return pairs receive zero
    weight. Future returns are loss targets, and are never passed to the
    prediction network.
    """
    if scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("Top-k pair loss requires aligned one-dimensional daily scores and labels")
    if not isinstance(topk, int) or topk < 1:
        raise ValueError("topk must be a positive integer")
    if len(scores) <= topk:
        return scores.sum()*0.0
    threshold = torch.topk(labels, topk, sorted=False).values.min()
    above, boundary = labels > threshold, labels == threshold
    fraction = (topk-above.sum()).to(scores.dtype)/boundary.sum()
    membership = above.to(scores.dtype)+boundary.to(scores.dtype)*fraction
    weights = membership[:, None]*(1-membership[None, :])
    weights = weights*(labels[:, None] > labels[None, :]).to(scores.dtype)
    gaps = scores[:, None]-scores[None, :]
    return (F.softplus(-gaps)*weights).sum()/weights.sum().clamp_min(1.0)
