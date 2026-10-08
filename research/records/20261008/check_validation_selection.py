"""Exercise frozen validation confirmation on six already completed seed triples."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from research import ARTIFACTS, ROOT
from research.common import code_fingerprint, config_id, digest_file, now, write_json
import research.selection as selection

DESTINATION = ROOT/"research/records/20261008"
NAMES = ["csi300_ple16_no_raw_seeds012",
         "csi300_frozen_mixture_1_regime_0_alpha_seeds012",
         "csi300_frozen_mixture_2_regime_0_alpha_seeds012",
         "sp500_raw_excess_seeds012", "sp500_independent_regime_seeds012",
         "sp500_frozen_mixture_1_regime_0.5_alpha_seeds012"]


def main():
    bundle = Path(selection.__file__).resolve().parent.parent
    assert bundle.parent == ARTIFACTS/"code_releases", "Preview must execute from an actual frozen package"
    assert code_fingerprint() == json.loads((bundle/"manifest.json").read_text())["files"]
    assert not (ARTIFACTS/"study/selection_lock.json").exists()
    assert not list((ARTIFACTS/"holdout").glob("*/test_run.json"))
    prior = json.loads((DESTINATION/"seed_validation_checks.json").read_text())["records"]
    controls = {row["name"]:row for row in prior if row["name"] in NAMES}
    assert set(controls) == set(NAMES)
    rows, evidence = [], {}
    for name in NAMES:
        source = controls[name]["individuals"][0]
        config = source["config"]
        assert config["seed"] == 0
        row = json.loads((ARTIFACTS/"trials"/config_id(config)/"result.json").read_text())
        # The production read_results adds the directory ID to result.json.
        rows.append({**row, "id":config_id(config)})
        for individual in controls[name]["individuals"]:
            folder = ARTIFACTS/"trials"/individual["id"]
            for relative in ["result.json", "run.json", "config.json", "valid/predictions.pkl", "best.pt"]:
                path = folder/relative
                if path.exists():
                    evidence[str(path)] = {"sha256":digest_file(path), "mtime_ns":path.stat().st_mtime_ns}
    plan_path = ARTIFACTS/"study/selection_preview_plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        assert plan["code"] == code_fingerprint()
    else:
        plan = selection.build_plan(rows, {"validation_shortlist_per_market":3,
            "promotion_models_per_market":1, "validation_seeds":[0,1,2], "seeds":[0,1,2,3,4]},
            bundle, namespace="validation_confirmation_preview")
        write_json(plan_path, plan)
    assert {item["id"] for items in plan["candidates"].values() for item in items} == {row["id"] for row in rows}
    state = {"completed":[], "failed":[], "phase":"validation_confirmation"}
    real_evaluate = selection.evaluate_predictions
    calls = []
    def evaluate(frame, market, output, backtest=True):
        started = time.monotonic()
        print(json.dumps({"event":"backtest_start", "market":market, "output":str(output)}, ensure_ascii=False), flush=True)
        result = real_evaluate(frame, market, output, backtest=backtest)
        calls.append(str(output))
        print(json.dumps({"event":"backtest_complete", "market":market, "seconds":time.monotonic()-started}, ensure_ascii=False), flush=True)
        return result
    selection.evaluate_predictions = evaluate
    def no_fit(*args, **kwargs):
        raise AssertionError("Preview must reuse complete three-seed fits")
    assert selection.confirm_plan(plan, state, no_fit, lambda state:None)
    first_calls = len(calls)
    confirmation = json.loads(Path(state["validation_confirmed"]["report"]).read_text())
    checks = []
    cached_files = {}
    for record in confirmation["records"]:
        source_ids = [source["id"] for source in record["sources"]]
        name = next(name for name, control in controls.items() if source_ids == [source["id"] for source in control["individuals"]])
        control = controls[name]
        old_output = Path(control.get("output", ARTIFACTS/"study/seed_validation"/name))
        actual = pd.read_pickle(Path(record["directory"])/"predictions.pkl")
        expected = pd.read_pickle(old_output/"predictions.pkl")
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        difference = max(abs(record["validation"][key]-control["ensemble"][key]) for key in selection.METRICS)
        assert difference <= 1e-12, (name, difference)
        assert record["validation"] == json.loads((Path(record["directory"])/"metrics.json").read_text())
        checks.append({"name":name, "id":record["id"], "source_ids":source_ids, "rows":record["rows"],
                       "exact_predictions_and_dtype":True, "max_metric_difference":difference,
                       "validation":record["validation"], "directory":record["directory"]})
        for path in Path(record["directory"]).iterdir():
            if path.is_file():
                cached_files[str(path)] = {"sha256":digest_file(path), "mtime_ns":path.stat().st_mtime_ns}
    restart = {"completed":[], "failed":[], "phase":"validation_confirmation"}
    assert selection.confirm_plan(plan, restart, no_fit, lambda state:None)
    assert len(calls) == first_calls, "Restart repeated an actual portfolio backtest"
    for name, proof in {**evidence, **cached_files}.items():
        path = Path(name)
        assert {"sha256":digest_file(path), "mtime_ns":path.stat().st_mtime_ns} == proof, name
    assert restart["validation_confirmed"]["selected"] == state["validation_confirmed"]["selected"]
    assert len(state["completed"]) == len(restart["completed"]) == 18
    assert not (ARTIFACTS/"study/selection_lock.json").exists()
    assert not (ARTIFACTS/"study/validation_confirmation_plan.json").exists()
    assert not list((ARTIFACTS/"holdout").glob("*/test_run.json"))
    proof = {"created_at":now(), "scope":"six representative existing three-seed validation ensembles; preview only; not the official shortlist or final selection",
             "executing_package":str(bundle/"research"), "code":code_fingerprint(), "helper_sha256":digest_file(Path(__file__)),
             "plan":plan, "checks":checks, "rankings":confirmation["rankings"], "preview_selected":confirmation["selected"],
             "actual_backtests_this_invocation":first_calls, "cached_restart_extra_backtests":len(calls)-first_calls,
             "reused_fits":18, "source_files_unchanged":len(evidence), "cached_files_unchanged_after_restart":len(cached_files),
             "no_refits":True, "no_official_lock_or_plan":True, "new_model_holdout_records":0}
    write_json(DESTINATION/"validation_selection_checks.json", proof)
    pd.DataFrame([{"name":row["name"], "id":row["id"], "rows":row["rows"], **row["validation"]} for row in checks]).to_csv(
        DESTINATION/"validation_selection_metrics.csv", index=False)
    print(json.dumps({"passed":True, "ensembles":len(checks), "max_metric_difference":max(row["max_metric_difference"] for row in checks),
                      "actual_backtests":first_calls, "restart_extra_backtests":0, "source_files_unchanged":len(evidence)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
