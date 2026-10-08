"""Aligned score combinations and causal, per-instrument signal smoothing."""
from __future__ import annotations

import numpy as np
import pandas as pd


def normalized_scores(frame, normalization):
    scores = frame["score"].astype(float)
    groups = scores.groupby(level="datetime")
    if normalization == "none":
        return scores
    if normalization == "cs_rank":
        return (groups.rank(pct=True)-0.5)*np.sqrt(12)
    if normalization == "cs_z":
        return (scores-groups.transform("mean"))/groups.transform("std",ddof=0).clip(lower=1e-6)
    raise ValueError(f"Unknown score normalization: {normalization}")


def causal_smoothing(scores, alpha=1.0, max_gap=5):
    """Update on observed dates only; reset after a gap of more than max_gap trading dates."""
    if not 0 < alpha <= 1 or max_gap < 1:
        raise ValueError("Invalid causal smoothing parameters")
    if alpha == 1.0:
        return scores.copy()
    wide = scores.unstack("instrument")
    values = wide.to_numpy()
    present = np.isfinite(values)
    output = np.full_like(values,np.nan,dtype=float)
    history = np.zeros(values.shape[1])
    last_seen = np.full(values.shape[1],-1,dtype=int)
    for day in range(len(values)):
        mask = present[day]
        recent = (last_seen[mask]>=0)&(day-last_seen[mask]<=max_gap)
        history[mask] = np.where(recent,alpha*values[day,mask]+(1-alpha)*history[mask],values[day,mask])
        output[day,mask] = history[mask]
        last_seen[mask] = day
    rows,columns = np.nonzero(present)
    index = pd.MultiIndex.from_arrays([wide.index.to_numpy()[rows],wide.columns.to_numpy()[columns]],
                                     names=scores.index.names)
    return pd.Series(output[present],index=index,name="score").reindex(scores.index)


def blend_predictions(frames, weights, normalization="cs_z", alpha=1.0, max_gap=5):
    if not frames or len(frames) != len(weights):
        raise ValueError("A score blend requires one weight per source")
    weights = np.asarray(weights,dtype=float)
    if not np.isfinite(weights).all() or (weights<0).any() or weights.sum()<=0:
        raise ValueError("Invalid score blend weights")
    weights /= weights.sum()
    reference = frames[0]
    if reference.index.names != ["datetime","instrument"] or not reference.index.is_unique or not reference.index.is_monotonic_increasing:
        raise ValueError("Score sources require sorted, unique datetime/instrument indices")
    result = reference.copy()
    combined = np.zeros(len(reference),dtype=float)
    for frame,weight in zip(frames,weights):
        if not frame.index.equals(reference.index):
            raise ValueError("Score source date/instrument coverage differs")
        np.testing.assert_equal(frame.label.to_numpy(),reference.label.to_numpy())
        if not np.isfinite(frame.score.to_numpy()).all():
            raise ValueError("Non-finite source scores")
        combined += weight*normalized_scores(frame,normalization).to_numpy()
    result["score"] = causal_smoothing(pd.Series(combined,index=reference.index),alpha,max_gap)
    return result
