"""Training-only past-beta residual labels; forecast inputs and evaluation stay raw."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import ARTIFACTS
from .common import config_id, digest_file, write_json, now
from .historical_risk import aligned_risk, rolling_price_risk

KIND = "market_residual_standardized"
TRAIN_END = pd.Timestamp("2020-12-31")


def residual_spec(config):
    strength = config.get("market_residual_strength", 1.0)
    window = config.get("market_beta_window", 120)
    minimum = config.get("market_beta_min_observations", 60)
    if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("Market removal strength must be finite and between zero and one")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in [window, minimum]) or not 2 <= minimum <= window:
        raise ValueError("Market beta requires integer 2 <= minimum <= window")
    return {"kind":KIND,"strength":float(strength),"window":window,"minimum":minimum,
            "beta_clip":[-3.,3.],"missing_beta":1.,"target":"raw stock return - strength * clipped past beta * forward benchmark return",
            "market_label":"close[t+5] / close[t+1] - 1; same horizon as stock baseline label",
            "beta":"paired finite daily simple returns; trailing OLS with intercept; prices through each forecast date",
            "data_scope":"selected training rows only; all forward benchmark endpoints no later than 2020-12-31",
            "prediction":"original features only; no market labels or beta cache queried for validation/test inference"}


def validate_training_index(index):
    if index.names != ["datetime","instrument"] or not index.is_unique or not index.is_monotonic_increasing or not len(index):
        raise ValueError("Market residual labels require a nonempty sorted unique training index")
    dates = index.get_level_values("datetime")
    if dates.min() < pd.Timestamp("2009-01-01") or dates.max() > TRAIN_END:
        raise ValueError("Market residual labels cannot query validation or test dates")


def forward_market_labels(prices, index):
    validate_training_index(index)
    if not prices.index.is_unique or not prices.index.is_monotonic_increasing or prices.index.max() > TRAIN_END:
        raise ValueError("Benchmark label prices must be unique sorted trading dates within training")
    prices = prices.astype(np.float64).where(lambda value:np.isfinite(value)&(value>0))
    forward = prices.shift(-5)/prices.shift(-1)-1
    result = forward.reindex(index.get_level_values("datetime")).to_numpy()
    if not np.isfinite(result).all():
        raise ValueError("Missing benchmark prices or incomplete five-day training horizon")
    return result


def market_residual_returns(raw_returns, market_returns, beta, strength):
    residual_spec({"market_residual_strength":strength})
    raw = np.asarray(raw_returns,dtype=np.float64)
    if raw.ndim != 1 or not len(raw) or not np.isfinite(raw).all():
        raise ValueError("Raw training returns must be a finite nonempty vector")
    if strength == 0:
        return raw.copy()
    market, exposure = np.asarray(market_returns,dtype=np.float64), np.asarray(beta,dtype=np.float64)
    if market.shape != raw.shape or exposure.shape != raw.shape or not np.isfinite(market).all():
        raise ValueError("Market target/exposure arrays must match selected training returns")
    exposure = np.where(np.isfinite(exposure),exposure,1.).clip(-3.,3.)
    return raw-strength*exposure*market


def training_market_data(data, config):
    if data.split != "train":
        raise ValueError("Market residual supervision is training-only")
    index = data.index[data.selected_positions()]
    validate_training_index(index)
    spec = residual_spec(config)
    dates = index.get_level_values("datetime")
    key = {"market":data.market,"forecast_index_sha256":hashlib.sha256(pd.util.hash_pandas_object(index,index=False).to_numpy().tobytes()).hexdigest(),
           "window":spec["window"],"minimum":spec["minimum"],"source_train_manifest_sha256":digest_file(ARTIFACTS/"cache"/data.market/"train/manifest.json"),
           "implementation_sha256":digest_file(Path(__file__)),
           "historical_estimator_sha256":digest_file(Path(__file__).with_name("historical_risk.py"))}
    folder = ARTIFACTS/"market_targets"/data.market/config_id(key)
    if (folder/"manifest.json").exists():
        manifest = json.loads((folder/"manifest.json").read_text())
        if manifest["key"] != key:
            raise ValueError("Training market cache specification has changed")
        for name, expected in manifest["files"].items():
            if digest_file(folder/name) != expected:
                raise ValueError("Training market cache has changed: "+name)
        features = pd.read_pickle(folder/"features.pkl")
        if not features.index.equals(index) or list(features.columns) != ["beta","observations","market_return"]:
            raise ValueError("Training market cache coverage differs")
        return features,{"path":str(folder),"manifest_sha256":digest_file(folder/"manifest.json"),**manifest}
    from .protocol import initialize_qlib,market_config
    from qlib.data import D
    initialize_qlib(data.market)
    calendar = pd.DatetimeIndex(D.calendar(start_time="2000-01-01",end_time=TRAIN_END))
    first,last = calendar.get_indexer([dates.min(),dates.max()])
    if first < spec["window"] or last < 0 or last+5 >= len(calendar):
        raise ValueError("Past beta history or purged within-training forward horizon is unavailable")
    start, label_end = calendar[first-spec["window"]],calendar[last+5]
    benchmark = market_config(data.market)["benchmark"]
    stocks = sorted(set(index.get_level_values("instrument")))
    history = D.features(stocks+[benchmark],["$close"],start_time=start,end_time=dates.max(),freq="day")
    history = history.iloc[:,0].unstack("instrument").reindex(calendar[(calendar>=start)&(calendar<=dates.max())])
    history = history.reindex(columns=stocks+[benchmark])
    benchmark_prices = D.features([benchmark],["$close"],start_time=start,end_time=label_end,freq="day")
    benchmark_prices = benchmark_prices.iloc[:,0].droplevel("instrument").reindex(calendar[(calendar>=start)&(calendar<=label_end)])
    estimates = rolling_price_risk(history,benchmark,spec["window"],spec["minimum"])
    features = aligned_risk({name:estimates[name] for name in ["beta","observations"]},index)
    features["market_return"] = forward_market_labels(benchmark_prices,index)
    folder.mkdir(parents=True,exist_ok=True)
    history.to_pickle(folder/"history_prices.pkl")
    benchmark_prices.to_pickle(folder/"benchmark_prices.pkl")
    features.to_pickle(folder/"features.pkl")
    manifest = {"created_at":now(),"key":key,"benchmark":benchmark,"past_price_start":str(start),
                "stock_price_end":str(dates.max()),"benchmark_label_price_end":str(label_end),
                "rows":len(index),"missing_beta_fraction":float(features.beta.isna().mean()),
                "roles":{"history_prices":"past-only beta supervision through each forecast date",
                         "benchmark_prices":"training forward-label construction only; never forecast input",
                         "features":"training target construction only; never forecast input"},
                "files":{name:digest_file(folder/name) for name in ["history_prices.pkl","benchmark_prices.pkl","features.pkl"]}}
    write_json(folder/"manifest.json",manifest)
    return features,{"path":str(folder),"manifest_sha256":digest_file(folder/"manifest.json"),**manifest}


def build_training_targets(data, config, raw_returns):
    from .train import standardized_return_targets
    if data.split != "train":
        raise ValueError("Market residual supervision cannot fit validation or test labels")
    index = data.index[data.selected_positions()]
    validate_training_index(index)
    raw = np.asarray(raw_returns,dtype=np.float32)
    if raw.ndim != 1 or len(raw) != len(index) or not np.isfinite(raw).all():
        raise ValueError("Selected raw training returns differ from training coverage")
    spec = residual_spec(config)
    metadata = None
    if spec["strength"] == 0:
        targets,scale = standardized_return_targets(raw,index,"raw_standardized")
    else:
        features,metadata = training_market_data(data,config)
        residual = market_residual_returns(raw,features.market_return,features.beta,spec["strength"])
        targets,scale = standardized_return_targets(residual,index,"raw_standardized")
    signature = {"spec":spec,"raw_return_sha256":hashlib.sha256(raw.tobytes()).hexdigest(),
                 "target_sha256":hashlib.sha256(targets.tobytes()).hexdigest(),"scale":scale,
                 "forecast_index_sha256":hashlib.sha256(pd.util.hash_pandas_object(index,index=False).to_numpy().tobytes()).hexdigest(),
                 "market_cache_manifest_sha256":metadata["manifest_sha256"] if metadata else None}
    proof = {**signature,"target_signature":hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest(),
             "market_data":metadata,"train_samples":len(index),"validation_labels":"original raw stock return; not residualized",
             "test_targets_used":False,"forecast_inputs_changed":False,"clipping":[-8,8]}
    return targets,scale,proof
