"""Past-price risk proxies applied to a frozen alpha under the baseline strategy."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import ARTIFACTS
from .common import code_fingerprint, config_id, digest_file, model_artifact_hashes, now, write_json
from .risk_regime import frozen_alpha_config
from .signals import normalized_scores


def risk_spec(config):
    window, minimum = config.get("risk_window", 120), config.get("risk_min_observations", 60)
    if any(isinstance(x, bool) or not isinstance(x, int) for x in [window, minimum]):
        raise ValueError("Historical risk windows must be integers")
    if not 2 <= minimum <= window:
        raise ValueError("Historical risk requires 2 <= minimum observations <= window")
    mode = config.get("historical_risk_mode", "beta")
    if mode not in ("beta", "total_volatility", "idiosyncratic_volatility"):
        raise ValueError("Unknown historical risk proxy")
    penalty = config.get("risk_penalty", 0.0)
    if isinstance(penalty, bool) or not isinstance(penalty, (int, float)) or not math.isfinite(penalty) or penalty < 0:
        raise ValueError("Historical risk penalty must be finite and nonnegative")
    normalization = config.get("alpha_norm", "cs_z")
    transform = config.get("risk_score_transform", "linear")
    if normalization not in ("cs_z", "native") or transform not in ("linear", "positive"):
        raise ValueError("Unknown historical alpha scale or risk score transform")
    spec = {"version":1, "window":window, "minimum_observations":minimum, "mode":mode,
            "penalty":float(penalty), "alpha_norm":normalization, "risk_z_clip":3.0,
            "market_variance_floor":1e-12, "volatility_log_floor":1e-8,
            "missing_risk":"zero standardized penalty; do not drop or fill stock-price observations",
            "zero_penalty":"return the original alpha frame exactly, preserving dtype and scores"}
    if normalization != "cs_z" or transform != "linear":
        spec.update(version=2, risk_score_transform=transform, native_score_scale_floor=1e-6,
                    native_score_scale="forecast-date alpha std(ddof=0); offsets and dispersion retained",
                    positive_risk="clip standardized risk below zero before subtracting penalty")
    return spec


def check_index(index):
    if index.names != ["datetime", "instrument"] or not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("Historical risk requires a sorted unique datetime/instrument index")


def rolling_price_risk(prices, benchmark, window=120, minimum=60):
    """OLS with intercept on paired finite daily simple returns, ending at date t."""
    risk_spec({"risk_window":window, "risk_min_observations":minimum})
    if not prices.index.is_unique or not prices.index.is_monotonic_increasing or not prices.columns.is_unique:
        raise ValueError("Prices require sorted unique dates and unique instruments")
    if benchmark not in prices.columns:
        raise ValueError("Missing market benchmark prices")
    values = prices.astype(np.float64).where(lambda x: np.isfinite(x) & (x > 0))
    returns = values.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    stocks = returns.drop(columns=benchmark)
    market = returns[benchmark]
    x = pd.DataFrame(np.broadcast_to(market.to_numpy()[:, None], stocks.shape), index=stocks.index, columns=stocks.columns)
    finite = stocks.notna() & x.notna()
    y, x = stocks.where(finite), x.where(finite)
    count = finite.astype(float).rolling(window, min_periods=1).sum()
    sx, sy = x.rolling(window, min_periods=minimum).sum(), y.rolling(window, min_periods=minimum).sum()
    sxx = (x*x).rolling(window, min_periods=minimum).sum()
    syy = (y*y).rolling(window, min_periods=minimum).sum()
    sxy = (x*y).rolling(window, min_periods=minimum).sum()
    var_x = ((sxx-sx*sx/count)/(count-1)).clip(lower=0)
    var_y = ((syy-sy*sy/count)/(count-1)).clip(lower=0)
    cov = (sxy-sx*sy/count)/(count-1)
    beta = (cov/var_x).where((count >= minimum) & (var_x > 1e-12))
    residual_variance = (var_y-cov*cov/var_x).clip(lower=0).where(beta.notna())
    return {"beta":beta, "total_volatility":np.sqrt(var_y),
            "idiosyncratic_volatility":np.sqrt(residual_variance), "observations":count}


def aligned_risk(risks, index):
    check_index(index)
    # Reindex preserves every baseline-eligible row, including unavailable prices.
    output = pd.DataFrame(index=index)
    for name, wide in risks.items():
        series = wide.stack(future_stack=True)
        series.index.names = ["datetime", "instrument"]
        output[name] = series.reindex(index)
    return output


def risk_z(features, spec):
    values = features[spec["mode"]].astype(np.float64).replace([np.inf, -np.inf], np.nan)
    if spec["mode"] != "beta":
        values = np.log(values.clip(lower=spec["volatility_log_floor"]))
    groups = values.groupby(level="datetime")
    center = groups.transform("mean")
    scale = groups.transform("std", ddof=0)
    z = (values-center)/scale.where(scale > 1e-12)
    return z.fillna(0.0).clip(-spec["risk_z_clip"], spec["risk_z_clip"])


def apply_historical_risk(frame, features, config):
    spec = risk_spec(config)
    check_index(frame.index)
    if not frame.index.equals(features.index) or not np.isfinite(frame.score.to_numpy()).all():
        raise ValueError("Historical risk/alpha alignment or finite scores differ")
    if spec["penalty"] == 0:
        return frame.copy()
    output = frame.copy()
    if spec["version"] == 1:
        output["score"] = normalized_scores(frame, "cs_z") - spec["penalty"]*risk_z(features, spec)
    else:
        from .risk_shaping import shaped_risk_scores
        output["score"] = shaped_risk_scores(frame, risk_z(features, spec), spec["penalty"],
                                            spec["alpha_norm"], spec["risk_score_transform"])
    return output


def load_price_risk(config, index, split):
    """Validation cache; holdout queries require the frozen selection/code lock."""
    check_index(index)
    spec = risk_spec(config)
    dates = index.get_level_values("datetime")
    if split == "test":
        from .train import require_locked_holdout
        require_locked_holdout(config)
    elif split != "valid" or dates.min() < pd.Timestamp("2021-01-01") or dates.max() > pd.Timestamp("2022-12-31"):
        raise ValueError("Historical price study may only query validation or locked holdout dates")
    from .protocol import initialize_qlib, market_config
    from qlib.data import D
    coverage = hashlib.sha256(pd.util.hash_pandas_object(index, index=False).to_numpy().tobytes()).hexdigest()
    key = {"market":config["market"], "split":split, "window":spec["window"], "minimum":spec["minimum_observations"],
           "coverage_sha256":coverage, "implementation_sha256":digest_file(Path(__file__))}
    identifier = config_id(key)
    folder = ARTIFACTS/"historical_risk"/identifier
    if (folder/"manifest.json").exists():
        manifest = json.loads((folder/"manifest.json").read_text())
        if manifest["key"] != key:
            raise ValueError("Historical risk cache specification has changed")
        for name, expected in manifest["files"].items():
            if digest_file(folder/name) != expected:
                raise ValueError("Historical risk cache has changed: "+name)
        features = pd.read_pickle(folder/"features.pkl")
        if not features.index.equals(index):
            raise ValueError("Historical risk cached coverage differs")
        return features, {"path":str(folder), "manifest_sha256":digest_file(folder/"manifest.json"), **manifest}
    initialize_qlib(config["market"])
    calendar = pd.DatetimeIndex(D.calendar(start_time="2009-01-01", end_time=dates.max()))
    first = calendar.searchsorted(dates.min())
    if first < spec["window"]:
        raise ValueError("Insufficient pre-split calendar history")
    start = calendar[first-spec["window"]]
    benchmark = market_config(config["market"])["benchmark"]
    instruments = sorted(set(index.get_level_values("instrument")))
    prices = D.features(instruments+[benchmark], ["$close"], start_time=start, end_time=dates.max(), freq="day")
    prices = prices.iloc[:, 0].unstack("instrument").reindex(calendar[calendar >= start])
    prices = prices.reindex(columns=instruments+[benchmark])
    features = aligned_risk(rolling_price_risk(prices, benchmark, spec["window"], spec["minimum_observations"]), index)
    folder.mkdir(parents=True, exist_ok=True)
    prices.to_pickle(folder/"prices.pkl")
    features.to_pickle(folder/"features.pkl")
    manifest = {"created_at":now(), "key":key, "benchmark":benchmark, "query_start":str(start), "query_end":str(dates.max()),
                "rows":len(features), "price_rows":len(prices), "instrument_count":len(instruments),
                "files":{name:digest_file(folder/name) for name in ["prices.pkl", "features.pkl"]},
                "features":"paired finite daily simple returns; trailing OLS intercept; sample variances ddof=1",
                "time_boundary":"only prices through forecast date; Qlib executes the signal on its internal next trading date",
                "labels_used":False, "price_fill":None,
                "available_fraction":{name:float(features[name].notna().mean()) for name in features.columns if name != "observations"}}
    write_json(folder/"manifest.json", manifest)
    return features, {"path":str(folder), "manifest_sha256":digest_file(folder/"manifest.json"), **manifest}


def train_historical_risk(config, destination):
    from .train import run_training
    risk_spec(config)
    source = frozen_alpha_config(config)
    identifier = config_id(source)
    trained = ARTIFACTS/"trials"/identifier
    if not (trained/"result.json").exists():
        print(f"FIT_HISTORICAL_ALPHA {identifier} seed={source['seed']}", flush=True)
        run_training(source, trained)
    result = json.loads((trained/"result.json").read_text())
    if result["config"] != source or result["status"] != "complete" or result.get("smoke"):
        raise ValueError("Historical risk requires a complete matching full alpha model")
    prediction = trained/"valid/predictions.pkl"
    frame = pd.read_pickle(prediction)
    if config.get("smoke_valid_days"):
        dates = frame.index.get_level_values("datetime").unique()[:config["smoke_valid_days"]]
        frame = frame.loc[frame.index.get_level_values("datetime").isin(dates)]
    features, cache = load_price_risk(config, frame.index, "valid")
    hashes = model_artifact_hashes(trained)
    if not hashes:
        raise FileNotFoundError("Historical alpha has no reproducible checkpoint")
    write_json(destination/"components.json", {"family":"historical_risk", "alpha":{"id":identifier, "config":source,
               "prediction_sha256":digest_file(prediction), "model_sha256":hashes}, "risk_spec":risk_spec(config), "validation_price_cache":cache,
               "alpha_seed":"follows the method seed", "risk_estimator":"deterministic past-price estimates shared across seeds",
               "selection":"validation only; unchanged baseline ranking and portfolio metrics"})
    write_json(destination/"historical_risk.json", {"risk_spec":risk_spec(config), "price_cache":cache,
               "missing_rows":int(features[risk_spec(config)["mode"]].isna().sum()), "labels_used_for_scoring":False,
               "backtest":"baseline Top30/drop5/risk_degree0.95/costs unchanged"})
    return apply_historical_risk(frame, features, config)


def verify_historical_source(config, trained):
    record = json.loads((Path(trained)/"components.json").read_text())
    source = frozen_alpha_config(config)
    if record["family"] != "historical_risk" or record["alpha"]["config"] != source or record["risk_spec"] != risk_spec(config):
        raise ValueError("Historical risk frozen component or specification differs")
    alpha = ARTIFACTS/"trials"/record["alpha"]["id"]
    if record["alpha"]["id"] != config_id(source) or model_artifact_hashes(alpha) != record["alpha"]["model_sha256"]:
        raise ValueError("Historical risk frozen alpha checkpoint has changed")
    cache = record["validation_price_cache"]
    folder = Path(cache["path"])
    if digest_file(folder/"manifest.json") != cache["manifest_sha256"]:
        raise ValueError("Historical risk validation price manifest has changed")
    for name, expected in cache["files"].items():
        if digest_file(folder/name) != expected:
            raise ValueError("Historical risk validation price cache has changed")
    return record


def predict_historical_risk(config, trained, destination):
    from .protocol import evaluate_predictions
    from .train import predict_test, require_locked_holdout
    require_locked_holdout(config)
    record = verify_historical_source(config, trained)
    source = record["alpha"]["config"]
    output = ARTIFACTS/"holdout"/record["alpha"]["id"]
    cached = json.loads((output/"test_run.json").read_text()) if (output/"test_run.json").exists() else {}
    expected = record["alpha"]["model_sha256"]
    matches = cached.get("config") == source and cached.get("trained_artifacts") == expected and cached.get("code") == code_fingerprint()
    if not (output/"metrics.json").exists() or not matches:
        predict_test(source, ARTIFACTS/"trials"/record["alpha"]["id"], output)
    frame = pd.read_pickle(output/"predictions.pkl")
    features, cache = load_price_risk(config, frame.index, "test")
    write_json(Path(destination)/"historical_risk.json", {"risk_spec":risk_spec(config), "price_cache":cache, "labels_used_for_scoring":False})
    return evaluate_predictions(apply_historical_risk(frame, features, config), config["market"], destination, backtest=True)
