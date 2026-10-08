"""Training-only parameter EMA; fitted feature buffers retain their source values."""
from __future__ import annotations

import math

import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn


def ema_decay(config):
    decay = config.get("ema_decay")
    if decay is None:
        return None
    if isinstance(decay, bool) or not isinstance(decay, (int, float)) or not math.isfinite(decay) or not 0 < decay < 1:
        raise ValueError("ema_decay must be a finite number strictly between zero and one")
    if config["family"] in ("ridge", "lgbm", "scores_blend") or config.get("alpha_source"):
        raise ValueError("EMA requires an optimized neural model; frozen scoring and tree/linear models do not update weights")
    return float(decay)


def make_ema(model, config):
    decay = ema_decay(config)
    if decay is None:
        return None
    # No current models use running BatchNorm. Numerical feature and regime buffers
    # must be copied from the training model, rather than averaged or refitted.
    if any(isinstance(module, torch.nn.modules.batchnorm._BatchNorm) for module in model.modules()):
        raise ValueError("EMA BatchNorm statistics need a separate training-only protocol")
    averaged = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay), use_buffers=False)
    averaged.requires_grad_(False)
    return averaged
