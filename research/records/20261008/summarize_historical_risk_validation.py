"""Matched past-price risk scoring, real-data audits and preserved negative results."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import pandas as pd

from research import ARTIFACTS, ROOT
from research.common import config_id, digest_file, model_artifact_hashes, now, write_json
from research.historical_risk import verify_historical_source


def summarize():
    study = ARTIFACTS/"study"
    destination = ROOT/"research/records/20261008"
    design = json.loads((study/"historical_risk_design.json").read_text())
    checks = {name:json.loads((study/(name+".json")).read_text())
              for name in ["historical_risk_cpu_checks", "historical_risk_route_checks"]}
    assert all(row["passed"] and row["code"] == design["code"] for row in checks.values())
    assert checks["historical_risk_route_checks"]["cases_completed"] == 2
    reuse = json.loads((study/"historical_risk_idempotency_checks.json").read_text())
    assert reuse["passed"] and reuse["checked_files"] == 80
    bundle = Path(design["bundle"])
    manifest = json.loads((bundle/"manifest.json").read_text())
    assert manifest["files"] == design["code"]
    assert all(digest_file(bundle/name) == expected for name, expected in manifest["files"].items())
    saved = destination/"code_snapshots"/bundle.name
    if not saved.exists():
        shutil.copytree(bundle, saved, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    records, pending = [], []
    for config in design["candidates"]:
        identifier = config_id(config)
        folder = ARTIFACTS/"trials"/identifier
        if not (folder/"result.json").exists():
            pending.append(identifier)
            continue
        result = json.loads((folder/"result.json").read_text())
        assert result["config"] == config and result["status"] == "complete" and not result.get("smoke")
        components = verify_historical_source(config, folder)
        run = json.loads((folder/"run.json").read_text())
        records.append({"id":identifier, "config":config, "validation":result["validation"],
                        "seconds":result["seconds"], "resources":result.get("resources"), "run":run,
                        "components":components, "components_sha256":digest_file(folder/"components.json"),
                        "prediction_sha256":digest_file(folder/"valid/predictions.pkl")})
    controls = {row["config"]["market"]:row for row in records if row["config"]["risk_penalty"] == 0}
    keys = ["IC", "ICIR", "RankIC", "RankICIR", "AR", "STD", "MDD", "Sharpe", "Sortino", "Calmar"]
    for row in records:
        control = controls.get(row["config"]["market"])
        if control:
            assert row["components"]["alpha"] == control["components"]["alpha"]
            row["matched_zero_penalty"] = {"id":control["id"], "deltas":{
                key:row["validation"][key]-control["validation"][key] for key in keys}}
    combinations = []
    from research.ensembles import source_configs
    for name in ["historical_risk_blend_design", "historical_learned_risk_blend_design"]:
        plan = json.loads((study/(name+".json")).read_text())
        complete = []
        for item in plan["comparisons"]:
            folder = ARTIFACTS/"trials"/item["id"]
            if not (folder/"result.json").exists():
                continue
            result = json.loads((folder/"result.json").read_text())
            assert result["config"] == item["config"] and result["status"] == "complete" and not result.get("smoke")
            components = json.loads((folder/"components.json").read_text())
            assert [source["config"] for source in components["sources"]] == source_configs(item["config"])
            for source in components["sources"]:
                fitted = ARTIFACTS/"trials"/source["id"]
                assert model_artifact_hashes(fitted) == source["model_sha256"]
                assert digest_file(fitted/"valid/predictions.pkl") == source["prediction_sha256"]
                if source["config"]["family"] == "historical_risk":
                    verify_historical_source(source["config"], fitted)
            complete.append({**item, "validation":result["validation"], "seconds":result["seconds"],
                             "run":json.loads((folder/"run.json").read_text()), "components":components,
                             "prediction_sha256":digest_file(folder/"valid/predictions.pkl")})
        combinations.append({"design":plan, "planned":len(plan["comparisons"]), "completed":len(complete), "experiments":complete})
    pd.DataFrame([{"id":row["id"], "name":row["name"], "market":row["config"]["market"], **row["validation"]}
                  for group in combinations for row in group["experiments"]]).to_csv(destination/"historical_risk_blend_validation_metrics.csv", index=False)
    write_json(destination/"historical_risk_validation_checks.json", {
        "created_at":now(), "scope":"complete purged validation; implementation audits excluded from search trial counts; test sealed",
        "planned_full_trials":len(design["candidates"]), "completed_full_trials":len(records), "pending":pending,
        "design":design, "checks":checks, "completed_trial_reuse_checks":reuse,
        "initial_trial_reuse_check":json.loads((study/"historical_risk_idempotency_checks_initial.json").read_text()),
        "initial_checks":{name:json.loads((study/(name+"_initial.json")).read_text()) for name in checks},
        "harnesses":{name:digest_file(ROOT/"research/tests"/name) for name in ["test_historical_risk.py", "check_historical_risk.py"]},
        "notes":["Risk features use only observed prices through the forecast date; no forward-return labels or future filling.",
                 "Past-price beta/volatility are deterministic and shared across seeds; frozen alpha follows each method seed.",
                 "Nonzero penalties normalize each alpha seed within a date; the final ensemble still uses unmodified baseline avg_none.",
                 "Zero penalty preserves original scores and dtype exactly, including close ties.",
                 "The BAB paper uses leverage and shorting; this study only adapts past-price risk scoring under baseline long-only Top30.",
                 "Original learn-row label eligibility and JKP release/revision timing caveats remain unverified.",
                 "An improved validation backtest is insufficient to establish success against the holdout baseline envelopes."],
        "holdout_success_established":False, "experiments":records, "combination_studies":combinations})
    pd.DataFrame([{"id":row["id"], "market":row["config"]["market"], "risk_mode":row["config"]["historical_risk_mode"],
                   "risk_penalty":row["config"]["risk_penalty"], "seed":row["config"]["seed"],
                   "seconds":row["seconds"], **row["validation"]} for row in records],
                 columns=["id", "market", "risk_mode", "risk_penalty", "seed", "seconds", *keys]).to_csv(destination/"historical_risk_validation_metrics.csv", index=False)
    print({"historical_risk_full_trials":len(records), "planned":len(design["candidates"]),
           "real_market_zero_controls":checks["historical_risk_route_checks"]["cases_completed"], "test_evaluated":False})


if __name__ == "__main__":
    summarize()
