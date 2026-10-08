"""Full-market identity controls and exact legacy scoring from actual risk caches."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research import ARTIFACTS
from research.common import code_fingerprint, config_id, digest_file, now, write_json
from research.historical_risk import apply_historical_risk, risk_spec, risk_z, verify_historical_source
from research.risk_shaping import shaped_risk_scores
from research.train import run_training


def main():
    study = ARTIFACTS/"study"
    design = json.loads((study/"risk_shaping_design.json").read_text())
    assert code_fingerprint() == design["code"]
    assert Path(__import__("research").__file__).parent == Path(design["bundle"])/"research"
    legacy = json.loads(Path(design["legacy_design"]).read_text())
    assert digest_file(design["legacy_design"]) == design["legacy_design_sha256"]
    before = {}
    def freeze_file(path):
        before[str(path)] = {"sha256":digest_file(path), "mtime_ns":path.stat().st_mtime_ns}
    for config in legacy["candidates"]:
        folder = ARTIFACTS/"trials"/config_id(config)
        for name in ["result.json", "run.json", "valid/predictions.pkl", "components.json"]:
            freeze_file(folder/name)
    for config in design["zero_controls"]:
        for seed in [0,1,2]:
            folder = ARTIFACTS/"trials"/config_id({**config["alpha_source"],"seed":seed})
            for name in ["best.pt", "result.json", "run.json", "valid/predictions.pkl"]:
                freeze_file(folder/name)
    legacy_cases = []
    for config in legacy["candidates"]:
        folder = ARTIFACTS/"trials"/config_id(config)
        record = verify_historical_source(config, folder)
        assert risk_spec(config) == record["risk_spec"]
        alpha = ARTIFACTS/"trials"/record["alpha"]["id"]
        frame = pd.read_pickle(alpha/"valid/predictions.pkl")
        features = pd.read_pickle(Path(record["validation_price_cache"]["path"])/"features.pkl")
        actual = apply_historical_risk(frame, features, config)
        pd.testing.assert_frame_equal(actual, pd.read_pickle(folder/"valid/predictions.pkl"), check_exact=True)
        legacy_cases.append({"id":config_id(config), "config":config, "predictions_and_dtype_exact":True,
                             "specification_exact":True})
    zero_cases, shape_cases = [], []
    keys = ["IC","ICIR","RankIC","RankICIR","AR","STD","MDD","Sharpe","Sortino","Calmar"]
    for config in design["zero_controls"]:
        destination = ARTIFACTS/"risk_shaping_proofs"/config_id(config)
        if not (destination/"result.json").exists():
            run_training(config, destination)
        record = verify_historical_source(config, destination)
        source = ARTIFACTS/"trials"/record["alpha"]["id"]
        actual = pd.read_pickle(destination/"valid/predictions.pkl")
        expected = pd.read_pickle(source/"valid/predictions.pkl")
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        metrics = json.loads((destination/"valid/metrics.json").read_text())
        old_metrics = json.loads((source/"valid/metrics.json").read_text())
        deltas = {key:metrics[key]-old_metrics[key] for key in keys}
        assert max(abs(value) for value in deltas.values()) <= 1e-12
        first, second = pd.read_csv(destination/"valid/curve.csv"), pd.read_csv(source/"valid/curve.csv")
        assert first.shape == second.shape and first.columns.equals(second.columns)
        assert first.iloc[:,0].equals(second.iloc[:,0])
        np.testing.assert_allclose(first.iloc[:,1:], second.iloc[:,1:], rtol=1e-12, atol=1e-12)
        cache = record["validation_price_cache"]
        old_control = next(case for case in legacy["zero_controls"] if case["market"]==config["market"])
        old_record = json.loads((ARTIFACTS/"trials"/config_id(old_control)/"components.json").read_text())
        for name in ["prices.pkl", "features.pkl"]:
            pd.testing.assert_frame_equal(pd.read_pickle(Path(cache["path"])/name),
                pd.read_pickle(Path(old_record["validation_price_cache"]["path"])/name), check_exact=True)
        features = pd.read_pickle(Path(cache["path"])/"features.pkl")
        for mode in ["beta", "total_volatility"]:
            z = risk_z(features, risk_spec({**config,"historical_risk_mode":mode}))
            for transform in ["linear", "positive"]:
                fixed = {**config,"historical_risk_mode":mode,"risk_penalty":.5,"risk_score_transform":transform}
                native = apply_historical_risk(expected, features, fixed)
                normalized = apply_historical_risk(expected, features, {**fixed,"alpha_norm":"cs_z"})
                np.testing.assert_array_equal(native.score.groupby(level="datetime").rank().to_numpy(),
                    normalized.score.groupby(level="datetime").rank().to_numpy())
                scores = expected.score.astype(np.float64)
                scale = scores.groupby(level="datetime").transform("std", ddof=0).clip(lower=1e-6)
                risk = z if transform=="linear" else z.clip(lower=0)
                np.testing.assert_allclose(native.score, scores-.5*scale*risk, rtol=0, atol=0)
                if transform=="positive":
                    assert (native.score <= scores).all()
                    np.testing.assert_array_equal(native.score[z<=0], scores[z<=0])
                poisoned = expected.assign(label=123456789.)
                np.testing.assert_array_equal(apply_historical_risk(poisoned, features, fixed).score, native.score)
                shape_cases.append({"market":config["market"], "mode":mode,"transform":transform,
                    "same_single_seed_ranks_between_scales":True,"independent_formula_exact":True,
                    "labels_not_used":True,"rows":len(native)})
        zero_cases.append({"market":config["market"],"config":config,"destination":str(destination),
            "predictions_and_dtype_exact":True,"full_curve_matches":True,"metric_deltas":deltas,
            "original_price_cache_frames_exact":True,"new_price_cache":cache})
        print({"market":config["market"],"zero_prediction_exact":True,
               "metric_difference_max":max(abs(value) for value in deltas.values())},flush=True)
    for name, proof in before.items():
        path = Path(name)
        assert {"sha256":digest_file(path),"mtime_ns":path.stat().st_mtime_ns} == proof
    write_json(study/"risk_shaping_route_checks.json", {"created_at":now(),"passed":True,"code":code_fingerprint(),
        "scope":"both markets complete purged validation; legacy score compatibility and new zero controls; no test",
        "legacy_cases":legacy_cases,"zero_cases":zero_cases,"shape_cases":shape_cases,
        "source_files_unchanged":len(before),"harness_sha256":digest_file(Path(__file__))})


if __name__ == "__main__":
    main()
