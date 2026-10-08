"""Run the prespecified CPU score grid and optional cached three-seed diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd

from research import ARTIFACTS, ROOT
from research.common import code_fingerprint, config_id, digest_file, model_artifact_hashes, now, write_json
from research.historical_risk import apply_historical_risk, verify_historical_source
from research.protocol import evaluate_predictions
from research.train import run_training


def folder_for(config):
    identifier = config_id(config)
    for root in [ARTIFACTS/"trials", ARTIFACTS/"risk_shaping_full_trials", ARTIFACTS/"risk_shaping_proofs"]:
        path = root/identifier
        if (path/"result.json").exists():
            return path
    return ARTIFACTS/"risk_shaping_full_trials"/identifier


def run_full(design):
    study = ARTIFACTS/"study"
    for name in ["risk_shaping_cpu_checks", "risk_shaping_route_checks"]:
        proof = json.loads((study/(name+".json")).read_text())
        assert proof["passed"] and proof["code"] == design["code"]
    records = []
    for config in design["candidates"]:
        destination = folder_for(config)
        try:
            if not (destination/"result.json").exists():
                run_training(config, destination)
            result = json.loads((destination/"result.json").read_text())
            run = json.loads((destination/"run.json").read_text())
            assert result["config"] == config and result["status"] == "complete" and not result.get("smoke")
            assert run["code"] == design["code"] and run["source_package"] == str(Path(design["bundle"])/"research")
            verify_historical_source(config, destination)
            records.append({"id":config_id(config), "config":config, "directory":str(destination),
                "files":{name:digest_file(destination/name) for name in ["config.json", "result.json", "run.json", "valid/predictions.pkl", "components.json"]}})
            write_json(study/"risk_shaping_full_trials.json", {"updated_at":now(), "complete":len(records)==len(design["candidates"]),
                "completed":len(records), "planned":len(design["candidates"]), "code":code_fingerprint(), "trials":records,
                "scope":"actual complete full validation score runs; two initial zero audits reused; no test"})
            print({"completed":len(records), "planned":len(design["candidates"]), "id":config_id(config)}, flush=True)
        except Exception:
            write_json(destination/"failure.json", {"at":now(), "config":config, "traceback":traceback.format_exc()})
            raise


def run_ensembles(design):
    keys = ["IC","ICIR","RankIC","RankICIR","AR","STD","MDD","Sharpe","Sortino","Calmar"]
    legacy = json.loads((ROOT/"research/records/20261008/historical_risk_seed_validation_checks.json").read_text())["records"]
    original = json.loads((ROOT/"research/records/20261008/seed_validation_checks.json").read_text())["records"]
    frames, sources, baseline_controls = {}, {}, {}
    for config in design["zero_controls"]:
        market = config["market"]
        frames[market], sources[market] = [], []
        for seed in design["alpha_seeds"]:
            alpha = {**config["alpha_source"], "seed":seed}
            folder = ARTIFACTS/"trials"/config_id(alpha)
            result = json.loads((folder/"result.json").read_text())
            assert result["config"] == alpha and result["status"] == "complete" and not result.get("smoke")
            frame = pd.read_pickle(folder/"valid/predictions.pkl")
            assert set(frame.index.get_level_values("datetime").year) == {2021,2022}
            if frames[market]:
                pd.testing.assert_index_equal(frame.index, frames[market][0].index)
                np.testing.assert_array_equal(frame.label.to_numpy(), frames[market][0].label.to_numpy())
            frames[market].append(frame)
            sources[market].append({"id":config_id(alpha), "config":alpha,
                "prediction_sha256":digest_file(folder/"valid/predictions.pkl"), "model_sha256":model_artifact_hashes(folder)})
        baseline_controls[market] = next(row for row in original if [source["id"] for source in row["individuals"]] == [source["id"] for source in sources[market]])
    evidence = {}
    for items in sources.values():
        for source in items:
            folder = ARTIFACTS/"trials"/source["id"]
            for name in ["best.pt", "result.json", "run.json", "valid/predictions.pkl"]:
                path = folder/name
                evidence[str(path)] = {"sha256":digest_file(path), "mtime_ns":path.stat().st_mtime_ns}
    records = []
    for config in design["candidates"]:
        folder = folder_for(config)
        component = verify_historical_source(config, folder)
        cache = component["validation_price_cache"]
        features = pd.read_pickle(Path(cache["path"])/"features.pkl")
        market = config["market"]
        scored = [apply_historical_risk(frame, features, {**config,"seed":seed}) for seed, frame in zip(design["alpha_seeds"],frames[market])]
        pd.testing.assert_frame_equal(scored[0], pd.read_pickle(folder/"valid/predictions.pkl"), check_exact=True)
        combined = scored[0].copy()
        combined.score = np.mean([frame.score.to_numpy() for frame in scored], axis=0)
        identifier = config_id(config)
        output = ARTIFACTS/"validation_checks/risk_shaping"/identifier
        identity = {"method":"avg_none", "alpha_seeds":design["alpha_seeds"], "config":config,
                    "code":code_fingerprint(), "sources":sources[market], "price_cache_sha256":cache["manifest_sha256"]}
        saved = json.loads((output/"ensemble.json").read_text()) if (output/"ensemble.json").exists() else {}
        if saved.get("provenance") == identity:
            for name, expected in saved["files"].items():
                assert digest_file(output/name) == expected
            pd.testing.assert_frame_equal(combined, pd.read_pickle(output/"predictions.pkl"), check_exact=True)
            metrics = json.loads((output/"metrics.json").read_text())
        else:
            metrics = evaluate_predictions(combined, market, output, backtest=True)
            write_json(output/"ensemble.json", {"created_at":now(),"provenance":identity,
                "files":{name:digest_file(output/name) for name in ["predictions.pkl","metrics.json","protocol.json","curve.csv","qlib_report.pkl","daily_ranking.csv"]}})
        if config["risk_penalty"] == 0:
            control = baseline_controls[market]
            assert max(abs(metrics[key]-control["ensemble"][key]) for key in keys) <= 1e-12
        else:
            control = next(row for row in legacy if row["market"]==market and row["candidate"]["historical_risk_mode"]==config["historical_risk_mode"]
                           and row["candidate"]["risk_penalty"]==config["risk_penalty"])
        records.append({"name":"risk_shaping_"+identifier+"_alpha_seeds012", "id":identifier, "config":config,
            "market":market, "method":"avg_none", "alpha_seeds":design["alpha_seeds"], "risk_estimator":"shared deterministic past prices",
            "ensemble":metrics, "sources":sources[market], "rows":len(combined), "output":str(output),
            "ensemble_prediction_sha256":digest_file(output/"predictions.pkl"), "ensemble_record_sha256":digest_file(output/"ensemble.json"),
            "matched_control":{"name":control["name"],"ensemble":control["ensemble"],"same_alpha_source_ids":True,
                "deltas":{key:metrics[key]-control["ensemble"][key] for key in keys}}})
        write_json(ARTIFACTS/"study/risk_shaping_seed_validation_checks.json", {"updated_at":now(), "code":code_fingerprint(),
            "scope":"38 fixed actual three-alpha-seed validation ensembles; deterministic past risks; not additional individual fits; no test",
            "completed":len(records),"planned":len(design["candidates"]),"records":records,"source_files_unchanged":False})
        print({"seed_ensembles_completed":len(records),"planned":len(design["candidates"]),"id":identifier},flush=True)
    for name, proof in evidence.items():
        path = Path(name)
        assert {"sha256":digest_file(path),"mtime_ns":path.stat().st_mtime_ns} == proof
    result = json.loads((ARTIFACTS/"study/risk_shaping_seed_validation_checks.json").read_text())
    result.update(source_files_unchanged=True, checked_source_files=len(evidence))
    write_json(ARTIFACTS/"study/risk_shaping_seed_validation_checks.json",result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensembles", action="store_true")
    args = parser.parse_args()
    design = json.loads((ARTIFACTS/"study/risk_shaping_design.json").read_text())
    assert code_fingerprint() == design["code"]
    assert Path(__import__("research").__file__).parent == Path(design["bundle"])/"research"
    if args.ensembles:
        run_ensembles(design)
    else:
        run_full(design)


if __name__ == "__main__":
    main()
