"""Per-date conditional Gaussian likelihood with diagonal plus low-rank covariance."""
from __future__ import annotations

import torch


def low_rank_gaussian_nll(mean, diagonal_log_variance, loadings, labels):
    """Return the per-stock joint NLL, omitting the constant log(2*pi)/2.

    Covariance is diag(exp(diagonal_log_variance)) + loadings @ loadings.T.
    Only a factor-rank-sized matrix is factorized. A posterior residual form of
    the quadratic avoids cancellation between two large Mahalanobis terms.
    Float64 is used here even when the network runs with mixed precision.
    """
    if mean.ndim!=1 or labels.shape!=mean.shape or diagonal_log_variance.shape!=mean.shape:
        raise ValueError("Invalid daily Gaussian mean, diagonal variance or label shape")
    if loadings.ndim!=2 or loadings.shape[0]!=len(mean) or not len(mean):
        raise ValueError("Invalid daily factor loadings")
    error = labels.double()-mean.double()
    log_diagonal = diagonal_log_variance.double()
    inverse_diagonal = torch.exp(-log_diagonal)
    factors = loadings.double()
    rank = factors.shape[1]
    if not rank:
        return 0.5*(error.square()*inverse_diagonal+log_diagonal).mean()
    precision = torch.eye(rank,dtype=error.dtype,device=error.device)+factors.T@(inverse_diagonal[:,None]*factors)
    # Positive diagonal noise and the identity term make precision positive definite.
    # cholesky_ex avoids an extra CUDA synchronization; the trainer checks finite loss.
    cholesky,_ = torch.linalg.cholesky_ex(precision,check_errors=False)
    rhs = factors.T@(inverse_diagonal*error)
    posterior_mean = torch.cholesky_solve(rhs[:,None],cholesky).squeeze(-1)
    residual = error-factors@posterior_mean
    quadratic = (residual.square()*inverse_diagonal).sum()+posterior_mean.square().sum()
    log_determinant = log_diagonal.sum()+2*cholesky.diagonal().log().sum()
    return 0.5*(quadratic+log_determinant)/len(mean)
