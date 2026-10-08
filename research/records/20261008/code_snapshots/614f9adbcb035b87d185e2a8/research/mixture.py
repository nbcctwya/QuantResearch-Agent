"""Conditional univariate Gaussian mixtures; labels enter likelihoods only."""
from __future__ import annotations

import math

import torch


def _parameters(means, log_variances, logits):
    if means.ndim != 2 or min(means.shape) < 1:
        raise ValueError("Mixture parameters must have nonempty [stocks, components] shape")
    if log_variances.shape != means.shape or logits.shape != means.shape:
        raise ValueError("Mixture parameter shapes differ")
    return means.double(), log_variances.double(), logits.double()


def gaussian_mixture_log_prob(means, log_variances, logits, targets, *, include_normalizer=True):
    """Stable per-stock log density, with float64 likelihood arithmetic."""
    means, log_variances, logits = _parameters(means, log_variances, logits)
    if targets.ndim != 1 or len(targets) != len(means):
        raise ValueError("Mixture targets must have [stocks] shape")
    error = targets.double()[:, None]-means
    component = -0.5*(log_variances+error.square()*(-log_variances).exp())
    if include_normalizer:
        component = component-0.5*math.log(2*math.pi)
    return torch.logsumexp(torch.log_softmax(logits, dim=1)+component, dim=1)


def gaussian_mixture_nll(means, log_variances, logits, targets):
    # Match existing Gaussian training objectives, which omit the constant normalizer.
    return -gaussian_mixture_log_prob(means, log_variances, logits, targets,
                                      include_normalizer=False).mean()


def gaussian_mixture_moments(means, log_variances, logits):
    """Return mean, within-component variance and between-component variance."""
    means, log_variances, logits = _parameters(means, log_variances, logits)
    weights = torch.softmax(logits, dim=1)
    mean = (weights*means).sum(1)
    within = (weights*log_variances.exp()).sum(1)
    between = (weights*(means-mean[:, None]).square()).sum(1)
    return mean, within, between


def gaussian_mixture_cdf(means, log_variances, logits, values):
    """Evaluate a value per stock or a [stocks, points] grid, without labels."""
    means, log_variances, logits = _parameters(means, log_variances, logits)
    if values.ndim not in (1, 2) or len(values) != len(means):
        raise ValueError("Mixture CDF values must have [stocks] or [stocks, points] shape")
    scalar = values.ndim == 1
    values = values[:, None] if scalar else values
    standardized = (values.double()[:, :, None]-means[:, None, :]) * (-0.5*log_variances[:, None, :]).exp()
    result = (torch.special.ndtr(standardized)*torch.softmax(logits, dim=1)[:, None, :]).sum(2)
    return result[:, 0] if scalar else result


def gaussian_mixture_quantiles(means, log_variances, logits, levels=(0.1, 0.5, 0.9)):
    """Invert the mixture CDF, rather than using a moment-matched normal interval."""
    means, log_variances, logits = _parameters(means, log_variances, logits)
    levels = torch.as_tensor(levels, dtype=torch.float64, device=means.device)
    if levels.ndim != 1 or not len(levels) or not torch.isfinite(levels).all() or not ((levels>0)&(levels<1)).all():
        raise ValueError("Mixture quantile levels must be finite and strictly inside (0, 1)")
    component_quantiles = means[:, None, :]+torch.exp(0.5*log_variances)[:, None, :]*torch.special.ndtri(levels)[None, :, None]
    # At the smallest component quantile all component CDFs are <= the level;
    # at the largest all are >= it. This brackets each mixture quantile.
    lower, upper = component_quantiles.amin(2), component_quantiles.amax(2)
    for _ in range(48):
        midpoint = 0.5*(lower+upper)
        below = gaussian_mixture_cdf(means, log_variances, logits, midpoint)<levels[None, :]
        lower = torch.where(below, midpoint, lower)
        upper = torch.where(below, upper, midpoint)
    return 0.5*(lower+upper)
