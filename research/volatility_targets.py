"""Past-risk normalized training supervision; baseline forecasts stay scalar."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import ARTIFACTS
from .common import config_id, digest_file, now, write_json
from .historical_risk import aligned_risk, rolling_price_risk
from .market_targets import (TRAIN_END, build_training_targets, market_residual_returns,
                             residual_spec, training_market_data, validate_training_index)

KIND = "past_volatility_standardized"


def volatility_spec(config):
    base = config.get("volatility_target_base", "raw")
    mode = config.get("volatility_target_mode", "total_volatility")
    power = config.get("volatility_target_power", 1.0)
    if base not in ("raw", "market_residual") or mode not in ("total_volatility", "idiosyncratic_volatility"):
        raise ValueError("Unknown volatility target base or past-risk estimator")
    if isinstance(power, bool) or not isinstance(power, (int, float)) or not math.isfinite(power) or not 0 <= power <= 1:
        raise ValueError("Volatility target power must be finite and between zero and one")
    market = residual_spec(config)
    return {"version":1,"kind":KIND,"base":base,"mode":mode,"power":float(power),
        "window":market["window"],"minimum":market["minimum"],
        "market_residual":market if base == "market_residual" else None,
        "floor_quantile":.05,"absolute_floor":1e-6,"missing_sigma":"selected-training positive median, then floor",
        "zero_sigma":"floor; zero is a valid historical volatility estimate",
        "target":"base stock return / bounded past daily volatility ** power; then selected-training std and clip +-8",
        "past_estimator":"paired finite stock/benchmark simple returns through forecast date; sample ddof1; intercept OLS residual variance for idiosyncratic risk",
        "floor_and_fallback_fit":"selected training estimates only; no validation or test fit",
        "prediction":"original model inputs only; no volatility cache or market labels at validation/test inference",
        "zero_power":"exact original base target and scale; bypass all volatility estimation and fitted floors"}


def fit_volatility_statistics(volatility):
    values = np.asarray(volatility,dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("Past training volatility must be a nonempty vector")
    positive = values[np.isfinite(values)&(values>0)]
    if not len(positive):
        raise ValueError("No positive finite training volatility for floor and fallback")
    floor = max(float(np.quantile(positive,.05)),1e-6)
    return {"floor_quantile":.05,"floor":floor,"positive_median":float(np.median(positive)),
        "training_rows":len(values),"positive_finite_rows":len(positive),
        "zero_rows":int(np.count_nonzero(values==0)),
        "missing_or_negative_rows":int(np.count_nonzero(~np.isfinite(values)|(values<0)))}


def normalize_base_returns(base_returns, volatility, power, statistics):
    volatility_spec({"volatility_target_power":power})
    base = np.asarray(base_returns,dtype=np.float64)
    if base.ndim != 1 or not len(base) or not np.isfinite(base).all():
        raise ValueError("Base training returns must be a finite nonempty vector")
    if power == 0:
        return base.copy()
    sigma = np.asarray(volatility,dtype=np.float64)
    if sigma.shape != base.shape:
        raise ValueError("Past volatility must match selected training coverage")
    floor, median = statistics["floor"], statistics["positive_median"]
    if not math.isfinite(floor) or floor < 1e-6 or not math.isfinite(median) or median <= 0:
        raise ValueError("Invalid training-fitted volatility floor or fallback")
    sigma = np.maximum(np.where(np.isfinite(sigma)&(sigma>=0),sigma,median),floor)
    return base/np.power(sigma,power)


def training_volatility_data(data, config):
    if data.split != "train":
        raise ValueError("Past-volatility supervision is training-only")
    index = data.index[data.selected_positions()]
    validate_training_index(index)
    spec = volatility_spec(config)
    _, parent = training_market_data(data,config)
    key = {"market":data.market,"parent_market_manifest_sha256":parent["manifest_sha256"],
        "implementation_sha256":digest_file(Path(__file__)),
        "historical_estimator_sha256":digest_file(Path(__file__).with_name("historical_risk.py"))}
    folder = ARTIFACTS/"volatility_targets"/data.market/config_id(key)
    if (folder/"manifest.json").exists():
        manifest = json.loads((folder/"manifest.json").read_text())
        if manifest["key"] != key:
            raise ValueError("Past-volatility training cache specification differs")
        for name, expected in manifest["files"].items():
            if digest_file(folder/name) != expected:
                raise ValueError("Past-volatility training cache has changed: "+name)
        features = pd.read_pickle(folder/"features.pkl")
        if not features.index.equals(index) or list(features.columns) != ["total_volatility","idiosyncratic_volatility"]:
            raise ValueError("Past-volatility training cache coverage differs")
        return features,{"path":str(folder),"manifest_sha256":digest_file(folder/"manifest.json"),
            "parent_market_data":parent,**manifest}
    history = pd.read_pickle(Path(parent["path"])/"history_prices.pkl")
    if history.index.max() != index.get_level_values("datetime").max() or history.index.max() > TRAIN_END:
        raise ValueError("Past-volatility history cannot contain prices after the last training forecast")
    estimates = rolling_price_risk(history,parent["benchmark"],spec["window"],spec["minimum"])
    features = aligned_risk({name:estimates[name] for name in ["total_volatility","idiosyncratic_volatility"]},index)
    folder.mkdir(parents=True,exist_ok=True)
    features.to_pickle(folder/"features.pkl")
    manifest = {"created_at":now(),"key":key,"rows":len(index),"stock_price_end":str(history.index.max()),
        "history_price_sha256":parent["files"]["history_prices.pkl"],
        "data_scope":"training estimates only, using parent history_prices.pkl; benchmark forward-label prices never enter volatility estimates",
        "files":{"features.pkl":digest_file(folder/"features.pkl")}}
    write_json(folder/"manifest.json",manifest)
    return features,{"path":str(folder),"manifest_sha256":digest_file(folder/"manifest.json"),
        "parent_market_data":parent,**manifest}


def build_volatility_targets(data, config, raw_returns):
    from .train import standardized_return_targets
    if data.split != "train":
        raise ValueError("Past-volatility supervision cannot fit validation or test labels")
    index = data.index[data.selected_positions()]
    validate_training_index(index)
    raw = np.asarray(raw_returns,dtype=np.float32)
    if raw.ndim != 1 or len(raw) != len(index) or not np.isfinite(raw).all():
        raise ValueError("Raw training returns differ from selected coverage")
    spec = volatility_spec(config)
    metadata = statistics = base_proof = None
    if spec["power"] == 0:
        if spec["base"] == "raw":
            targets,scale = standardized_return_targets(raw,index,"raw_standardized")
        else:
            targets,scale,base_proof = build_training_targets(data,config,raw)
    else:
        features,metadata = training_volatility_data(data,config)
        if spec["base"] == "raw":
            base = raw.astype(np.float64)
        else:
            market = pd.read_pickle(Path(metadata["parent_market_data"]["path"])/"features.pkl")
            if not market.index.equals(index):
                raise ValueError("Market residual and past volatility training coverage differ")
            base = market_residual_returns(raw,market.market_return,market.beta,spec["market_residual"]["strength"])
        sigma = features[spec["mode"]].to_numpy(dtype=np.float64)
        statistics = fit_volatility_statistics(sigma)
        adjusted = normalize_base_returns(base,sigma,spec["power"],statistics)
        targets,scale = standardized_return_targets(adjusted,index,"raw_standardized")
    signature = {"spec":spec,"raw_return_sha256":hashlib.sha256(raw.tobytes()).hexdigest(),
        "target_sha256":hashlib.sha256(targets.tobytes()).hexdigest(),"scale":scale,
        "forecast_index_sha256":hashlib.sha256(pd.util.hash_pandas_object(index,index=False).to_numpy().tobytes()).hexdigest(),
        "volatility_cache_manifest_sha256":metadata["manifest_sha256"] if metadata else None,
        "volatility_statistics":statistics,"zero_power_base_signature":base_proof["target_signature"] if base_proof else None}
    proof = {**signature,"target_signature":hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest(),
        "volatility_data":metadata,"zero_power_base_proof":base_proof,"train_samples":len(index),
        "validation_labels":"original raw stock return; never risk-scaled","test_targets_used":False,
        "forecast_inputs_changed":False,"forward_market_labels_used":spec["base"]=="market_residual" and (spec["market_residual"]["strength"]>0),
        "clipping":[-8,8]}
    return targets,scale,proof
