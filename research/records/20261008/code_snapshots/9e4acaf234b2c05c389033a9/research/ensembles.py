"""Reproducible signal mixtures; source models and weights are part of the configuration."""
from __future__ import annotations

import json

import pandas as pd

from . import ARTIFACTS
from .common import code_fingerprint, config_id, digest_file, model_artifact_hashes, write_json
from .signals import blend_predictions


def source_configs(config):
    sources = []
    for source in config["sources"]:
        if source["family"] == "scores_blend" or source["market"] != config["market"]:
            raise ValueError("Score sources must be individual models from the same market")
        if source.get("purge_days",5) != config.get("purge_days",5):
            raise ValueError("Score sources have different validation boundary policies")
        sources.append({**source,"seed":config["seed"]})
    if not sources or len(sources) != len(config["weights"]):
        raise ValueError("Score source/weight size mismatch")
    return sources


def combine(config,frames):
    return blend_predictions(frames,config["weights"],config.get("score_norm","cs_z"),
                             config.get("ewm_alpha",1.0),config.get("max_gap",5))


def train_blend(config,destination):
    # This import is deferred because run_training also dispatches to this function.
    from .train import run_training
    frames,manifest = [],[]
    for source in source_configs(config):
        identifier = config_id(source)
        trained = ARTIFACTS/"trials"/identifier
        if not (trained/"result.json").exists():
            print(f"FIT_COMPONENT {identifier} seed={source['seed']}",flush=True)
            run_training(source,trained)
        result = json.loads((trained/"result.json").read_text())
        if result["config"] != source or result["status"] != "complete":
            raise ValueError("Completed score source configuration mismatch")
        if result.get("smoke") and not (config.get("smoke_train_days") or config.get("smoke_valid_days")):
            raise ValueError("A complete score mixture cannot use smoke-trained sources")
        prediction = trained/"valid/predictions.pkl"
        frame = pd.read_pickle(prediction)
        if config.get("smoke_valid_days"):
            dates = frame.index.get_level_values("datetime").unique()[:config["smoke_valid_days"]]
            frame = frame.loc[frame.index.get_level_values("datetime").isin(dates)]
        frames.append(frame)
        hashes = model_artifact_hashes(trained)
        if not hashes:
            raise FileNotFoundError("Completed score source has no reproducible model artifact")
        manifest.append({"id":identifier,"config":source,"prediction_sha256":digest_file(prediction),
                         "model_sha256":hashes})
    write_json(destination/"components.json",{
        "sources":manifest,"weights":config["weights"],"score_norm":config.get("score_norm","cs_z"),
        "ewm_alpha":config.get("ewm_alpha",1.0),"max_gap":config.get("max_gap",5),
        "selection":"validation only; each component uses the mixture's seed",
        "smoothing":"causal within this split; starts afresh on its first date; resets after long instrument gaps"})
    return combine(config,frames)


def predict_blend(config,trained,destination):
    from .protocol import evaluate_predictions
    from .train import predict_test
    manifest = json.loads((trained/"components.json").read_text())
    if [row["config"] for row in manifest["sources"]] != source_configs(config):
        raise ValueError("Frozen mixture component configurations differ")
    for key,default in [("weights",None),("score_norm","cs_z"),("ewm_alpha",1.0),("max_gap",5)]:
        if manifest[key] != config.get(key,default):
            raise ValueError("Frozen mixture scoring parameters differ")
    frames = []
    for source,row in zip(source_configs(config),manifest["sources"]):
        source_trained = ARTIFACTS/"trials"/row["id"]
        for name,expected in row["model_sha256"].items():
            if digest_file(source_trained/name) != expected:
                raise ValueError("A frozen mixture component checkpoint has changed")
        output = ARTIFACTS/"holdout"/row["id"]
        cached = json.loads((output/"test_run.json").read_text()) if (output/"test_run.json").exists() else {}
        current_code = code_fingerprint()
        cache_matches = cached.get("config") == source and cached.get("trained_artifacts") == row["model_sha256"]
        cache_matches = cache_matches and all(cached.get("code",{}).get(name)==current_code[name] for name in
                                             ["research/models.py","research/data.py","research/train.py","research/protocol.py"])
        if not (output/"metrics.json").exists() or not cache_matches:
            predict_test(source,source_trained,output)
        frames.append(pd.read_pickle(output/"predictions.pkl"))
    return evaluate_predictions(combine(config,frames),config["market"],destination,backtest=True)
