"""Bounded alpha-dispersion conditioning fitted on frozen training forecasts."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import ARTIFACTS
from .common import config_id, digest_file, write_json
from .risk_regime import RiskRegimeScaler, frozen_alpha_config


def daily_alpha_statistic(alpha):
    """Log within-date forecast dispersion; this is a hypothesis, not known quality."""
    if alpha.ndim != 1 or not len(alpha):
        raise ValueError("Alpha opportunity requires nonempty single-date scores")
    return alpha.float().std(correction=0).clamp_min(1e-6).log()


class AlphaOpportunityScaler(RiskRegimeScaler):
    def __init__(self, strength=0.5, temperature=1.0):
        if not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("alpha_opportunity_strength must be finite and between zero and one")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("alpha_opportunity_temperature must be finite and positive")
        super().__init__(strength, temperature)

    def forward(self, alpha):
        if not self.is_fitted:
            raise ValueError("Alpha opportunity statistics must be fitted on training dates or loaded")
        z = (daily_alpha_statistic(alpha)-self.center)/(self.scale*self.temperature)
        return 1-self.strength*torch.tanh(z)


def fit_frozen_alpha_opportunity(model, training, config, destination, amp):
    scaler = getattr(model, "opportunity", None)
    if scaler is None:
        return
    if not getattr(model, "freeze_alpha", False):
        raise ValueError("Alpha opportunity requires a frozen alpha source")
    if training.split != "train":
        raise ValueError("Alpha opportunity statistics may only be fitted on selected training dates")
    if scaler.is_fitted:
        return
    if model.alpha.training:
        raise ValueError("Training forecasts require the frozen alpha network in eval mode")
    from .models import prediction_scores
    statistics = []
    with torch.no_grad():
        for day in training.day_ids:
            stock, context = training.inputs(day)
            with torch.autocast(training.device.type, dtype=torch.bfloat16,
                                enabled=amp and training.device.type == "cuda"):
                alpha = prediction_scores(model.alpha(stock, context))
            statistics.append(daily_alpha_statistic(alpha))
    values = torch.stack(statistics).cpu().numpy().astype(np.float64)
    fit = scaler.fit(values)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination/"training_alpha_statistics.npy", values)
    source = frozen_alpha_config(config)
    write_json(destination/"alpha_opportunity.json", {
        "fit_split":"selected purged training features only", "days":len(training.day_ids),
        "first_training_date":str(training.dates[training.day_ids].min()),
        "last_training_date":str(training.dates[training.day_ids].max()),
        "training_day_ids_sha256":hashlib.sha256(training.day_ids.tobytes()).hexdigest(),
        "training_statistics_sha256":hashlib.sha256(values.tobytes()).hexdigest(),
        "labels_used":False, "validation_or_test_features_used":False,
        "statistic":"log(max(within-date population standard deviation of frozen alpha scores, 1e-6))",
        "multiplier":"1 - strength * tanh((statistic - training_median)/(training_IQR_scale * temperature))",
        "strength":scaler.strength, "temperature":scaler.temperature,
        "multiplier_bounds":[1-scaler.strength, 1+scaler.strength], **fit,
        "alpha_source":source,
        "alpha_source_checkpoint_sha256":digest_file(ARTIFACTS/"trials"/config_id(source)/"best.pt"),
        "hypothesis":"larger forecast dispersion may represent stronger differentiated opportunities; not established signal quality",
        "scoring":"alpha - risk_penalty * risk_regime_multiplier * alpha_opportunity_multiplier * standardized log_variance",
        "backtest":"unchanged baseline TopkDropoutStrategy, risk_degree and costs"})


def record_alpha_opportunity_diagnostics(model, data, destination, amp):
    if getattr(model, "opportunity", None) is None:
        return
    if data.split != "valid":
        raise ValueError("Research opportunity diagnostics only accept validation features")
    from .models import prediction_scores
    rows = []
    with torch.no_grad():
        for day in data.day_ids:
            stock, context = data.inputs(day)
            with torch.autocast(data.device.type, dtype=torch.bfloat16,
                                enabled=amp and data.device.type == "cuda"):
                alpha = prediction_scores(model.alpha(stock, context))
                risk_multiplier = model.regime(model.risk_model(stock, context)["log_variance"]) if model.regime is not None else alpha.new_tensor(1.)
            opportunity_multiplier = model.opportunity(alpha)
            rows.append(torch.stack([daily_alpha_statistic(alpha), opportunity_multiplier, risk_multiplier]))
    frame = pd.DataFrame(torch.stack(rows).cpu().numpy(),
                         columns=["alpha_statistic", "opportunity_multiplier", "risk_regime_multiplier"],
                         index=pd.DatetimeIndex(data.dates[data.day_ids], name="datetime"))
    frame["total_multiplier"] = frame.opportunity_multiplier*frame.risk_regime_multiplier
    frame["risk_penalty"] = frame.total_multiplier*model.risk_penalty
    destination = Path(destination)/"valid/alpha_opportunity"
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination/"daily_penalty.csv")
    write_json(destination/"summary.json", {
        "split":"purged validation features only", "days":len(frame), "labels_used":False,
        "mean_opportunity_multiplier":float(frame.opportunity_multiplier.mean()),
        "minimum_opportunity_multiplier":float(frame.opportunity_multiplier.min()),
        "maximum_opportunity_multiplier":float(frame.opportunity_multiplier.max()),
        "mean_total_multiplier":float(frame.total_multiplier.mean()),
        "mean_risk_penalty":float(frame.risk_penalty.mean()),
        "by_year":[{"year":int(year), "days":len(part),
                    "mean_opportunity_multiplier":float(part.opportunity_multiplier.mean()),
                    "mean_total_multiplier":float(part.total_multiplier.mean()),
                    "mean_risk_penalty":float(part.risk_penalty.mean())}
                   for year, part in frame.groupby(frame.index.year)],
        "parameters":"checkpoint buffers fitted only on selected frozen-alpha training-feature forecasts"})
