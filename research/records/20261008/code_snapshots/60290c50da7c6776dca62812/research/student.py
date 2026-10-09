"""Fixed-tail Student-t likelihood parameterized by finite conditional variance."""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import t
import torch


def validate_student_df(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 2:
        raise ValueError("student_df must be finite and greater than two for a finite scoring variance")
    return float(value)


def student_log_prob(mean, log_variance, target, degrees_of_freedom):
    if mean.ndim != 1 or not len(mean) or log_variance.shape != mean.shape or target.shape != mean.shape:
        raise ValueError("Student-t mean, log variance and target must be aligned nonempty vectors")
    if not isinstance(degrees_of_freedom, torch.Tensor):
        degrees_of_freedom = validate_student_df(degrees_of_freedom)
    df = torch.as_tensor(degrees_of_freedom, dtype=torch.float64, device=mean.device)
    if df.ndim != 0 and df.shape != mean.shape:
        raise ValueError("Student-t degrees of freedom must be scalar or aligned with forecasts")
    # CUDA tensors come from the model's constructor-validated fixed-df buffer.
    # Avoid a host synchronization for every training-date loss.
    if df.device.type == "cpu" and not (torch.isfinite(df).all() and (df > 2).all()):
        raise ValueError("Student-t scoring requires degrees of freedom greater than two")
    mean, log_variance, target = mean.double(), log_variance.double(), target.double()
    residual_square = (target - mean).square() * torch.exp(-log_variance)
    normalizer = torch.lgamma((df + 1) / 2) - torch.lgamma(df / 2) - 0.5 * torch.log((df - 2) * math.pi)
    return normalizer - 0.5 * log_variance - (df + 1) / 2 * torch.log1p(residual_square / (df - 2))


def student_nll(mean, log_variance, target, degrees_of_freedom):
    return -student_log_prob(mean, log_variance, target, degrees_of_freedom).mean()


def student_quantiles(mean, log_variance, degrees_of_freedom, levels=(0.1, 0.5, 0.9)):
    df = validate_student_df(degrees_of_freedom)
    probabilities = np.asarray(levels, dtype=np.float64)
    if probabilities.ndim != 1 or not len(probabilities) or not np.isfinite(probabilities).all() or not ((probabilities > 0) & (probabilities < 1)).all():
        raise ValueError("Student-t quantile probabilities must lie strictly between zero and one")
    if mean.ndim != 1 or log_variance.shape != mean.shape:
        raise ValueError("Student-t quantile parameters must be aligned vectors")
    standard_quantiles = torch.as_tensor(t.ppf(probabilities, df), dtype=torch.float64, device=mean.device)
    scale = torch.exp(0.5 * log_variance.double()) * math.sqrt((df - 2) / df)
    return mean.double()[:, None] + scale[:, None] * standard_quantiles


def student_cdf(mean, log_variance, value, degrees_of_freedom):
    """CPU observation diagnostic; never called while producing stock scores."""
    df = validate_student_df(degrees_of_freedom)
    mean, log_variance, value = [item.detach().double().cpu().numpy() for item in (mean, log_variance, value)]
    scale = np.exp(0.5 * log_variance) * math.sqrt((df - 2) / df)
    return t.cdf((value - mean) / scale, df)
