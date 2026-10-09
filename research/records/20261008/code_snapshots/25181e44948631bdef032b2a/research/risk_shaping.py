"""Past-risk score shaping that separates risk direction from alpha seed scale."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .signals import normalized_scores


def shaped_risk_scores(frame, standardized_risk, penalty, normalization="native", transform="positive"):
    """Only forecast-date scores/risk enter; labels are never consulted.

    Native units keep score offsets and dispersion for baseline avg_none.
    The within-date population standard deviation converts the dimensionless
    risk penalty to alpha units, with the existing normalization's 1e-6 floor.
    Positive shaping penalizes above-average risk without rewarding low risk.
    """
    if normalization not in {"native", "cs_z"} or transform not in {"linear", "positive"}:
        raise ValueError("Unknown historical alpha scale or risk score transform")
    if isinstance(penalty, bool) or not isinstance(penalty, (int, float)) or not math.isfinite(penalty) or penalty < 0:
        raise ValueError("Risk shaping penalty must be finite and nonnegative")
    if (frame.index.names != ["datetime", "instrument"] or not frame.index.is_unique
            or not frame.index.is_monotonic_increasing or not frame.index.equals(standardized_risk.index)):
        raise ValueError("Risk shaping requires aligned sorted unique forecast-date indices")
    if not np.isfinite(frame.score.to_numpy()).all() or not np.isfinite(standardized_risk.to_numpy()).all():
        raise ValueError("Risk shaping requires finite scores and neutralized risk values")
    if penalty == 0:
        return frame.score.copy()
    risk = standardized_risk.astype(np.float64)
    if transform == "positive":
        risk = risk.clip(lower=0)
    if normalization == "cs_z":
        return (normalized_scores(frame, "cs_z")-penalty*risk).rename("score")
    scores = frame.score.astype(np.float64)
    scale = scores.groupby(level="datetime").transform("std", ddof=0).clip(lower=1e-6)
    return (scores-penalty*scale*risk).rename("score")
