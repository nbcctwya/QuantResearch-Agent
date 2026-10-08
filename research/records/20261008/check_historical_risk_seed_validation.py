"""Actual three-alpha-seed averages with matched normalization-only controls."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research import ARTIFACTS, ROOT
from research.common import code_fingerprint, config_id, digest_file, model_artifact_hashes, now, write_json
from research.historical_risk import apply_historical_risk, load_price_risk, verify_historical_source
from research.protocol import evaluate_predictions
from research.risk_regime import frozen_alpha_config
from research.signals import normalized_scores

DESTINATION = ROOT/"research/records/20261008"
METRICS = ["IC", "ICIR", "RankIC", "RankICIR", "AR", "STD", "MDD", "Sharpe", "Sortino", "Calmar"]


def main():
    design = json.loads((ARTIFACTS/"study/historical_risk_design.json").read_text())
    code = code_fingerprint()
    assert code == design["code"]
    # Normalization-only controls already exist for these exact alpha sources.
    # Reuse after verifying predictions and unchanged scoring/backtest sources.
    scale = json.loads((DESTINATION/"score_scale_validation_checks.json").read_text())["records"]
    controls, source_frames, source_metadata = {}, {}, {}
    for method in design["zero_controls"]:
        market = method["market"]
        frames, metadata = [], []
        for seed in [0, 1, 2]:
            config = frozen_alpha_config({**method, "seed":seed})
            folder = ARTIFACTS/"trials"/config_id(config)
            result = json.loads((folder/"result.json").read_text())
            assert result["config"] == config and result["status"] == "complete" and not result.get("smoke")
            frame = pd.read_pickle(folder/"valid/predictions.pkl")
            assert set(frame.index.get_level_values("datetime").year) == {2021, 2022}
            if frames:
                pd.testing.assert_index_equal(frame.index, frames[0].index)
                np.testing.assert_equal(frame.label.to_numpy(), frames[0].label.to_numpy())
            frames.append(frame)
            metadata.append({"id":config_id(config), "config":config,
                             "prediction_sha256":digest_file(folder/"valid/predictions.pkl"),
                             "model_sha256":model_artifact_hashes(folder), "validation":result["validation"],
                             "run":json.loads((folder/"run.json").read_text())})
        normalized = frames[0].copy()
        normalized["score"] = np.mean([normalized_scores(frame, "cs_z").to_numpy() for frame in frames], axis=0)
        control = next(row for row in scale if row["market"] == market and row["normalization_inside_candidate"] == "cs_z"
                       and [source["id"] for source in row["sources"]] == [source["id"] for source in metadata])
        for key in ["research/signals.py", "research/protocol.py"]:
            assert control["evaluation_code"][key] == code[key]
        for actual, expected in zip(metadata, control["sources"]):
            assert actual["prediction_sha256"] == expected["prediction_sha256"]
        output = Path(control["output"])
        assert digest_file(output/"predictions.pkl") == control["prediction_sha256"]
        pd.testing.assert_frame_equal(normalized, pd.read_pickle(output/"predictions.pkl"), check_exact=True)
        assert json.loads((output/"metrics.json").read_text()) == control["validation"]
        protocol = json.loads((output/"protocol.json").read_text())
        assert digest_file(ROOT/protocol["metrics_source"]) == protocol["metrics_sha256"]
        assert digest_file(ROOT/protocol["backtest_source"]) == protocol["backtest_source_sha256"]
        controls[market] = control
        source_frames[market], source_metadata[market] = frames, metadata
    records, pending = [], []
    for candidate in design["candidates"]:
        identifier = config_id(candidate)
        folder = ARTIFACTS/"trials"/identifier
        if not (folder/"result.json").exists():
            pending.append(identifier)
            continue
        component = verify_historical_source(candidate, folder)
        market = candidate["market"]
        sources = source_metadata[market]
        frames = source_frames[market]
        features, cache = load_price_risk(candidate, frames[0].index, "valid")
        scored = [apply_historical_risk(frame, features, {**candidate, "seed":seed}) for seed, frame in enumerate(frames)]
        pd.testing.assert_frame_equal(scored[0], pd.read_pickle(folder/"valid/predictions.pkl"), check_exact=True)
        assert component["alpha"]["model_sha256"] == sources[0]["model_sha256"]
        combined = frames[0].copy()
        combined["score"] = np.mean([frame.score.to_numpy() for frame in scored], axis=0)
        mode, penalty = candidate["historical_risk_mode"], candidate["risk_penalty"]
        name = f"{market}_historical_{mode}_penalty_{penalty:g}_alpha_seeds012"
        output = ARTIFACTS/"validation_checks"/name
        provenance = {"method":"avg_none", "candidate":candidate, "alpha_seeds":[0, 1, 2],
                      "code":code, "helper_sha256":digest_file(Path(__file__)),
                      "source_predictions":[{"id":source["id"], "prediction":source["prediction_sha256"], "model":source["model_sha256"]} for source in sources],
                      "price_cache_manifest":cache["manifest_sha256"]}
        saved = json.loads((output/"ensemble.json").read_text()) if (output/"ensemble.json").exists() else {}
        if saved.get("provenance") == provenance and (output/"metrics.json").exists():
            assert digest_file(output/"predictions.pkl") == saved["prediction_sha256"]
            assert digest_file(output/"metrics.json") == saved["metrics_sha256"]
            pd.testing.assert_frame_equal(combined, pd.read_pickle(output/"predictions.pkl"), check_exact=True)
            metrics = json.loads((output/"metrics.json").read_text())
        else:
            metrics = evaluate_predictions(combined, market, output, backtest=True)
            write_json(output/"ensemble.json", {"provenance":provenance, "prediction_sha256":digest_file(output/"predictions.pkl"),
                       "metrics_sha256":digest_file(output/"metrics.json")})
        control = controls[market]
        deltas = {key:metrics[key]-control["validation"][key] for key in METRICS}
        records.append({"name":name, "created_at":now(), "market":market, "method":"avg_none", "candidate":candidate,
                        "seeds":[0, 1, 2], "alpha_seeds":[0, 1, 2], "risk_estimator":"shared deterministic past prices",
                        "scope":"purged validation stability diagnostic; no extra individual fits or averaged portfolio metrics; no holdout",
                        "rows":len(combined), "ensemble":metrics, "individuals":sources,
                        "normalization_only_control":{"name":control["name"], "validation":control["validation"],
                            "prediction_sha256":control["prediction_sha256"], "same_sources":True,
                            "deltas_risk_minus_normalization_only":deltas},
                        "ensemble_prediction_sha256":digest_file(output/"predictions.pkl"), "code":code, "output":str(output)})
        write_json(DESTINATION/"historical_risk_seed_validation_checks.json", {
            "created_at":now(), "scope":"three paired alpha seeds; deterministic past-price risk; test sealed",
            "planned_ensembles":len(design["candidates"]), "completed_ensembles":len(records), "records":records, "pending":pending,
            "notes":["Each seed is scored first and actual predictions are averaged; portfolio metrics are computed on the mean signal.",
                     "Normalization-only controls have the same checkpoints and exact recomputed scores; their scoring/backtest source hashes are checked.",
                     "Individual metadata denotes original alpha models; individual risk backtests are not fabricated or counted as new full trials.",
                     "Both alpha normalization and risk penalties differ from the raw zero-penalty ensemble; matched normalization-only controls isolate risk penalties."],
            "holdout_success_established":False})
        pd.DataFrame([{"name":row["name"], "market":row["market"], "risk_mode":row["candidate"]["historical_risk_mode"],
                       "risk_penalty":row["candidate"]["risk_penalty"], **row["ensemble"]} for row in records]).to_csv(DESTINATION/"historical_risk_seed_validation_metrics.csv", index=False)
        existing_path = DESTINATION/"seed_validation_checks.json"
        existing = json.loads(existing_path.read_text())
        if name not in {row["name"] for row in existing["records"]}:
            existing["records"].append(records[-1])
            existing["updated_at"] = now()
            write_json(existing_path, existing)
        print({"completed":name, "validation":metrics}, flush=True)
    print({"completed_ensembles":len(records), "pending":len(pending)})


if __name__ == "__main__":
    main()
