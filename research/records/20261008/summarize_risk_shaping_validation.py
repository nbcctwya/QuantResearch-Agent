"""Export the fixed risk-shaping grid with genuine completed and pending evidence."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from research import ARTIFACTS, ROOT
from research.common import config_id, digest_file, now, write_json
from research.historical_risk import verify_historical_source

KEYS = ["IC","ICIR","RankIC","RankICIR","AR","STD","MDD","Sharpe","Sortino","Calmar"]


def summarize():
    destination = ROOT/"research/records/20261008"
    study = ARTIFACTS/"study"
    design = json.loads((study/"risk_shaping_design.json").read_text())
    checks = {name:json.loads((study/(name+".json")).read_text()) for name in ["risk_shaping_cpu_checks","risk_shaping_route_checks"]}
    assert all(check["passed"] and check["code"]==design["code"] for check in checks.values())
    bundle = Path(design["bundle"])
    manifest = json.loads((bundle/"manifest.json").read_text())
    assert manifest["files"]==design["code"]
    assert all(digest_file(bundle/name)==expected for name,expected in manifest["files"].items())
    saved = destination/"code_snapshots"/bundle.name
    if not saved.exists():
        shutil.copytree(bundle,saved,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    proof = json.loads((study/"risk_shaping_full_trials.json").read_text())
    rows = []
    for item in proof["trials"]:
        original = Path(item["directory"])
        folder = original if original.exists() else ARTIFACTS/"trials"/item["id"]
        for name, expected in item["files"].items():
            assert digest_file(folder/name)==expected
        result = json.loads((folder/"result.json").read_text())
        run = json.loads((folder/"run.json").read_text())
        assert result["config"]==item["config"] and result["status"]=="complete" and not result.get("smoke")
        assert run["status"]=="complete" and run["code"]==design["code"]
        assert run["source_package"]==str(bundle/"research")
        components = verify_historical_source(item["config"],folder)
        rows.append({**item,"current_directory":str(folder),"validation":result["validation"],
                     "seconds":result["seconds"],"resources":result.get("resources"),"run":run,"components":components})
    assert len(rows)==proof["completed"]
    completed = {row["id"] for row in rows}
    pending = [config_id(config) for config in design["candidates"] if config_id(config) not in completed]
    legacy = json.loads(Path(design["legacy_design"]).read_text())
    affine_checks = []
    for row in rows:
        config = row["config"]
        if config["risk_penalty"]==0 or config["alpha_norm"]!="native" or config["risk_score_transform"]!="linear":
            continue
        control_config = next(item for item in legacy["candidates"] if item["market"]==config["market"]
            and item["historical_risk_mode"]==config["historical_risk_mode"] and item["risk_penalty"]==config["risk_penalty"])
        control = ARTIFACTS/"trials"/config_id(control_config)
        current = pd.read_pickle(Path(row["current_directory"])/"valid/predictions.pkl")
        old = pd.read_pickle(control/"valid/predictions.pkl")
        pd.testing.assert_index_equal(current.index,old.index)
        np.testing.assert_array_equal(current.score.groupby(level="datetime").rank(),old.score.groupby(level="datetime").rank())
        original_metrics = json.loads((control/"valid/metrics.json").read_text())
        difference = max(abs(row["validation"][key]-original_metrics[key]) for key in KEYS)
        assert difference<=1e-12
        affine_checks.append({"id":row["id"],"control":config_id(control_config),"exact_single_seed_ranks":True,"max_metric_difference":difference})
    seeds_path = study/"risk_shaping_seed_validation_checks.json"
    seeds = json.loads(seeds_path.read_text()) if seeds_path.exists() else {"completed":0,"planned":38,"records":[],"source_files_unchanged":False}
    if seeds["records"]:
        assert seeds["code"]==design["code"]
        for row in seeds["records"]:
            output=Path(row["output"])
            assert digest_file(output/"ensemble.json")==row["ensemble_record_sha256"]
            cache=json.loads((output/"ensemble.json").read_text())
            for name,expected in cache["files"].items():
                assert digest_file(output/name)==expected
    write_json(destination/"risk_shaping_validation_checks.json",{
        "created_at":now(),"scope":"fixed purged-validation score grid; GPU search continues independently; no holdout",
        "design":design,"checks":checks,"install":json.loads((study/"risk_shaping_install.json").read_text()),
        "jobs":json.loads((study/"risk_shaping_jobs.json").read_text()),"planned_full_runs":38,"completed_full_runs":len(rows),
        "pending":pending,"full_grid_complete":proof["complete"],"experiments":rows,
        "native_linear_single_seed_affine_controls":affine_checks,"three_seed_diagnostics":seeds,
        "notes":["Each completed row is a genuine full score run with baseline backtest, not a new neural fit.",
                 "Two complete zero-penalty implementation audits are reused as the same grid IDs; they are not counted twice.",
                 "Temporary CPU outputs enter the coordinator only after the full grid and seed follower finish and import verifies all hashes.",
                 "Native and normalized linear penalties give the same within-seed ranks but different seed weights in avg_none.",
                 "Positive shaping penalizes high risk and does not grant a low-risk bonus.",
                 "No topk, dropout, position, execution or fee settings change; this does not replicate portfolio volatility targeting.",
                 "Validation improvements do not establish success against holdout baseline envelopes.",
                 "The inherited learn-row DropnaLabel and JKP release/revision timing caveats remain."],
        "holdout_success_established":False,"new_model_holdout_records":len(list((ARTIFACTS/"holdout").glob("*/test_run.json")))})
    pd.DataFrame([{"id":row["id"],"market":row["config"]["market"],"risk_mode":row["config"]["historical_risk_mode"],
        "penalty":row["config"]["risk_penalty"],"alpha_norm":row["config"]["alpha_norm"],"risk_transform":row["config"]["risk_score_transform"],
        "seconds":row["seconds"],**row["validation"]} for row in rows]).to_csv(destination/"risk_shaping_validation_metrics.csv",index=False)
    pd.DataFrame([{"id":row["id"],"market":row["market"],"risk_mode":row["config"]["historical_risk_mode"],
        "penalty":row["config"]["risk_penalty"],"alpha_norm":row["config"]["alpha_norm"],"risk_transform":row["config"]["risk_score_transform"],
        **row["ensemble"]} for row in seeds["records"]],columns=["id","market","risk_mode","penalty","alpha_norm","risk_transform",*KEYS]).to_csv(
        destination/"risk_shaping_seed_validation_metrics.csv",index=False)
    shutil.copy2(study/"risk_shaping_cpu_checks.log",destination/"risk_shaping_cpu_checks.log")
    print({"full_score_runs":len(rows),"planned":38,"three_seed_ensembles":seeds["completed"],
           "native_affine_controls":len(affine_checks),"cpu_checks":116,"test_evaluated":False})


if __name__ == "__main__":
    summarize()
