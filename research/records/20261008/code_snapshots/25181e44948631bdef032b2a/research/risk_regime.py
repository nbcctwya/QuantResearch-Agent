"""Training-fitted forecast-risk scaling without future returns as inputs."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from . import ARTIFACTS
from .common import config_id, digest_file, write_json


def frozen_alpha_config(config):
    source = config["alpha_source"]
    neural = {"residual", "temporal_mixer", "batch_ensemble", "risk_aware",
              "factor_gaussian", "mixture_gaussian", "quantile_aware", "student_t", "master_control"}
    if source["family"] not in neural or source["market"] != config["market"]:
        raise ValueError("Frozen alpha requires a same-market individual neural source")
    if source.get("purge_days",5) != config.get("purge_days",5):
        raise ValueError("Frozen alpha and its scorer have different validation boundary policies")
    return {**source, "seed":config["seed"]}


def verify_frozen_sources(config, trained):
    if config["family"] == "historical_risk":
        from .historical_risk import verify_historical_source
        verify_historical_source(config, trained)
        return
    if not config.get("alpha_source"):
        return
    alpha = json.loads((Path(trained)/"fixed_alpha.json").read_text())
    if alpha["config"] != frozen_alpha_config(config):
        raise ValueError("Frozen alpha manifest configuration differs")
    risk = json.loads((Path(trained)/"risk_source.json").read_text())
    if risk["config"] != config["risk_source"]:
        raise ValueError("Frozen risk manifest configuration differs")
    for name, record in [("alpha", alpha), ("risk", risk)]:
        source = ARTIFACTS/"trials"/config_id(record["config"])/"best.pt"
        if digest_file(source) != record["checkpoint_sha256"]:
            raise ValueError("Frozen "+name+" checkpoint has changed")


def daily_risk_statistic(log_variance):
    """Log arithmetic mean of this date's predicted marginal variances."""
    if log_variance.ndim != 1 or not len(log_variance):
        raise ValueError("Risk regime requires a nonempty single-date variance forecast")
    return torch.logsumexp(log_variance.float(), 0)-math.log(len(log_variance))


class RiskRegimeScaler(nn.Module):
    def __init__(self, strength=0.5, temperature=1.0):
        super().__init__()
        if not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("risk_regime_strength must be finite and between zero and one")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("risk_regime_temperature must be finite and positive")
        self.strength, self.temperature = strength, temperature
        self.register_buffer("center", torch.tensor(0.0))
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("fitted", torch.tensor(False))
        self.is_fitted = False
        self.register_load_state_dict_post_hook(self._loaded)

    def _loaded(self, module, incompatible_keys):
        self.is_fitted = bool(self.fitted.cpu().item())

    def fit(self, statistics):
        values = np.asarray(statistics, dtype=np.float64)
        if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
            raise ValueError("Risk regime requires at least two finite training-date statistics")
        center = float(np.median(values))
        q25, q75 = np.quantile(values, [0.25, 0.75])
        scale = max(float((q75-q25)/1.3489795003921634), 0.1)
        self.center.fill_(center)
        self.scale.fill_(scale)
        self.fitted.fill_(True)
        self.is_fitted = True
        return {"center_median":center, "scale_iqr_over_1_34898":scale,
                "minimum_scale":0.1, "training_q25":float(q25), "training_q75":float(q75)}

    def forward(self, log_variance):
        if not self.is_fitted:
            raise ValueError("Risk regime statistics must be fitted on training dates or loaded from a checkpoint")
        z = (daily_risk_statistic(log_variance)-self.center)/(self.scale*self.temperature)
        return 1+self.strength*torch.tanh(z)


def fit_frozen_risk_regime(model, training, config, destination, amp):
    scaler = getattr(model, "regime", None)
    if scaler is None:
        return
    if training.split != "train":
        raise ValueError("Risk regime statistics may only be fitted on selected training dates")
    if scaler.is_fitted:
        return
    statistics = []
    with torch.no_grad():
        for day in training.day_ids:
            stock, context = training.inputs(day)
            with torch.autocast(training.device.type, dtype=torch.bfloat16,
                                enabled=amp and training.device.type == "cuda"):
                log_variance = model.risk_model(stock, context)["log_variance"]
            statistics.append(daily_risk_statistic(log_variance))
    values = torch.stack(statistics).cpu().numpy().astype(np.float64)
    fit = scaler.fit(values)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination/"training_risk_statistics.npy", values)
    write_json(destination/"risk_regime.json", {
        "fit_split":"selected purged training dates only", "days":len(training.day_ids),
        "first_training_date":str(training.dates[training.day_ids].min()),
        "last_training_date":str(training.dates[training.day_ids].max()),
        "training_day_ids_sha256":hashlib.sha256(training.day_ids.tobytes()).hexdigest(),
        "training_statistics_sha256":hashlib.sha256(values.tobytes()).hexdigest(),
        "labels_used":False, "validation_or_test_features_used":False,
        "statistic":"log(mean(exp(frozen forecast log_variance))) per date",
        "multiplier":"1 + strength * tanh((statistic - training_median)/(training_IQR_scale * temperature))",
        "strength":scaler.strength, "temperature":scaler.temperature,
        "multiplier_bounds":[1-scaler.strength, 1+scaler.strength], **fit,
        "risk_source":config["risk_source"],
        "risk_source_checkpoint_sha256":digest_file(ARTIFACTS/"trials"/config_id(config["risk_source"])/"best.pt"),
        "scoring":"alpha - risk_penalty * multiplier * within-date standardized log_variance",
        "backtest":"unchanged baseline TopkDropoutStrategy, risk_degree and costs"})


def record_risk_regime_diagnostics(model, data, destination, amp):
    if getattr(model, "regime", None) is None:
        return
    if data.split != "valid":
        raise ValueError("Research regime diagnostics only accept validation features")
    rows = []
    with torch.no_grad():
        for day in data.day_ids:
            stock, context = data.inputs(day)
            with torch.autocast(data.device.type, dtype=torch.bfloat16,
                                enabled=amp and data.device.type == "cuda"):
                log_variance = model.risk_model(stock, context)["log_variance"]
            rows.append(torch.stack([daily_risk_statistic(log_variance), model.regime(log_variance)]))
    frame = pd.DataFrame(torch.stack(rows).cpu().numpy(), columns=["risk_statistic", "multiplier"],
                         index=pd.DatetimeIndex(data.dates[data.day_ids], name="datetime"))
    frame["risk_penalty"] = frame.multiplier*model.risk_penalty
    destination = Path(destination)/"valid/risk_regime"
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination/"daily_penalty.csv")
    records = [{"year":int(year), "days":len(part),
                "mean_multiplier":float(part.multiplier.mean()),
                "mean_risk_penalty":float(part.risk_penalty.mean())}
               for year, part in frame.groupby(frame.index.year)]
    write_json(destination/"summary.json", {
        "split":"purged validation features only", "days":len(frame), "labels_used":False,
        "mean_multiplier":float(frame.multiplier.mean()),
        "minimum_multiplier":float(frame.multiplier.min()), "maximum_multiplier":float(frame.multiplier.max()),
        "by_year":records, "parameters":"checkpoint buffers fitted on selected training dates"})
