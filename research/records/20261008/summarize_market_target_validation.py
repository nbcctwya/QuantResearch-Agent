"""Export fixed training-target controls and their genuine completed evidence."""
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd
import torch

from research import ARTIFACTS, ROOT
from research.common import config_id, digest_file, now, write_json
from research.selection import verify_confirmation_record


def main():
    study = ARTIFACTS/"study"
    destination = ROOT/"research/records/20261008"
    design = json.loads((study/"market_residual_design.json").read_text())
    bundle = Path(design["bundle"])
    assert digest_file(bundle/"manifest.json") == design["code_manifest_sha256"]
    assert all(digest_file(bundle/name) == value for name,value in design["code"].items())
    checks = {name:json.loads((study/(name+".json")).read_text())
              for name in ["market_residual_data_checks","market_residual_gpu_checks"]}
    assert all(value["passed"] and value["code"] == design["code"] for value in checks.values())
    parents = []
    for config in design["architecture_parents"].values():
        folder = ARTIFACTS/"trials"/config_id(config)
        result = json.loads((folder/"result.json").read_text())
        assert result["config"] == config and result["status"] == "complete" and not result.get("smoke")
        parents.append({"id":config_id(config),"config":config,"validation":result["validation"],
            "run":json.loads((folder/"run.json").read_text()),
            "files":{name:digest_file(folder/name) for name in ["result.json","run.json","best.pt","valid/predictions.pkl"]}})
    rows, pending = [], []
    for item in design["candidates"]:
        config, identifier = item["config"], item["id"]
        assert config_id(config) == identifier
        folder = ARTIFACTS/"trials"/identifier
        if not (folder/"result.json").exists():
            pending.append(identifier)
            continue
        result = json.loads((folder/"result.json").read_text())
        run = json.loads((folder/"run.json").read_text())
        if run["status"] != "complete":
            pending.append(identifier)
            continue
        assert result["config"] == config and result["status"] == "complete" and not result.get("smoke")
        if not item["preexisting_complete"]:
            assert run["code"] == design["code"] and run["source_package"] == str(bundle/"research")
        metadata = json.loads((folder/"target_transform.json").read_text())
        evidence = {"target_transform":metadata}
        if config["target_kind"] == "market_residual_standardized":
            proof = json.loads((folder/"market_residual.json").read_text())
            assert proof["spec"]["strength"] == config["market_residual_strength"]
            assert not proof["forecast_inputs_changed"] and not proof["test_targets_used"]
            signature = {name:proof[name] for name in ["spec","raw_return_sha256","target_sha256","scale",
                "forecast_index_sha256","market_cache_manifest_sha256"]}
            assert hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest() == proof["target_signature"]
            checkpoint = torch.load(folder/"last.pt",map_location="cpu",weights_only=False)
            assert checkpoint["training_target_signature"] == proof["target_signature"]
            market_data = proof["market_data"]
            cache = Path(market_data["path"])
            assert digest_file(cache/"manifest.json") == market_data["manifest_sha256"]
            assert all(digest_file(cache/name) == value for name,value in market_data["files"].items())
            assert pd.Timestamp(market_data["benchmark_label_price_end"]) <= pd.Timestamp("2020-12-31")
            assert market_data["key"]["implementation_sha256"] == design["code"]["research/market_targets.py"]
            evidence["market_residual"] = proof
        rows.append({**item,"run":run,"completed_at":result["completed_at"],"seconds":result["seconds"],
            "resources":result.get("resources"),"validation":result["validation"],"metadata":evidence,
            "files":{name:digest_file(folder/name) for name in ["config.json","run.json","result.json",
                "best.pt","valid/predictions.pkl","target_transform.json"]}})
    saved = destination/"code_snapshots"/bundle.name
    if not saved.exists():
        shutil.copytree(bundle,saved,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    cpu = {"passed":True,"tests":126,"log_sha256":digest_file(study/"market_residual_cpu_checks.log"),
           "code":design["code"]}
    seed_study = None
    seed_path = study/"market_residual_seed_design.json"
    if seed_path.exists():
        seed_design = json.loads(seed_path.read_text())
        assert seed_design["code"] == design["code"]
        seed_ready, seed_pending = [], []
        for item in seed_design["candidates"]:
            folder = ARTIFACTS/"trials"/item["id"]
            if (folder/"result.json").exists() and json.loads((folder/"run.json").read_text())["status"] == "complete":
                result = json.loads((folder/"result.json").read_text())
                assert result["config"] == item["config"] and result["status"] == "complete" and not result.get("smoke")
                seed_ready.append({**item,"validation":result["validation"],
                    "files":{name:digest_file(folder/name) for name in ["result.json","run.json","best.pt","valid/predictions.pkl"]}})
            else:
                seed_pending.append(item["id"])
        seed_checks_path = study/"market_residual_seed_validation_checks.json"
        seed_checks = json.loads(seed_checks_path.read_text()) if seed_checks_path.exists() else None
        seed_study = {"design":seed_design,"design_sha256":digest_file(seed_path),"ready":seed_ready,
            "pending":seed_pending,"completed_source_fits":len(seed_ready),"planned_source_fits":24,
            "extra_seed_queue":json.loads((study/"market_residual_seed_queue.json").read_text()),
            "follower":json.loads((study/"market_residual_seed_follower.json").read_text()),
            "jobs":json.loads((study/"market_residual_seed_jobs.json").read_text()),"actual_seed_ensembles":seed_checks}
        if seed_checks and seed_checks["completed"] == seed_checks["planned"] == 8 and seed_checks["source_files_unchanged"]:
            existing_path = destination/"seed_validation_checks.json"
            existing = json.loads(existing_path.read_text())
            names = {row["name"]:row for row in existing["records"]}
            repeated = []
            for record in seed_checks["records"]:
                verify_confirmation_record(record)
                if record["config"]["market"] == "sp500" and record["config"]["target_kind"] == "raw_excess_standardized":
                    old = names["sp500_raw_excess_seeds012"]
                    assert [source["id"] for source in record["sources"]] == [source["id"] for source in old["individuals"]]
                    original = Path(old.get("output",ARTIFACTS/"study/seed_validation"/old["name"]))
                    pd.testing.assert_frame_equal(pd.read_pickle(Path(record["directory"])/"predictions.pkl"),
                        pd.read_pickle(original/"predictions.pkl"),check_exact=True)
                    difference=max(abs(record["validation"][key]-old["ensemble"][key])
                        for key in ["IC","ICIR","RankIC","RankICIR","AR","STD","MDD","Sharpe","Sortino","Calmar"])
                    assert difference <= 1e-12
                    repeated.append({"name":record["name"],"original":old["name"],
                        "predictions_and_dtype_exact":True,"max_metric_difference":difference,
                        "counted_as_new_global_method_record":False})
                    continue
                summary={**record,"market":record["config"]["market"],"candidate":record["config"],
                    "ensemble":record["validation"],"output":record["directory"],"code":seed_checks["code"],
                    "scope":"actual matched three-seed target validation; unchanged baseline backtest; no test"}
                if record["name"] in names:
                    assert names[record["name"]] == summary
                else:
                    existing["records"].append(summary)
            existing["updated_at"]=now()
            write_json(existing_path,existing)
            seed_study["existing_control_reproduction"]=repeated
    failures = {name:digest_file(study/name) for name in ["market_residual_data_checks_initial.log",
        "market_residual_data_checks_float32_reference.log"]}
    write_json(destination/"market_residual_validation_checks.json",{
        "created_at":now(),"scope":"fixed matched training-target study; purged validation only; no new-model test",
        "design":design,"cpu_checks":cpu,"checks":checks,
        "install":json.loads((study/"market_residual_install.json").read_text()),
        "initial_check_failures":failures,
        "initial_failure_note":"First harness invocation omitted the frozen package from PYTHONPATH; a later independent OLS reference retained float32 close. Both exited, their logs remain; import environment and reference dtype corrected, error tolerances unchanged.",
        "planned_full_configs":8,"planned_new_full_fits":7,"completed_full_configs":len(rows),
        "new_full_fits_completed":sum(not row["preexisting_complete"] for row in rows),
        "pending":pending,"experiments":rows,"architecture_reference_controls":parents,"seed_study":seed_study,
        "architecture_reference_note":"Existing CSI300 CSRank and SP500 raw-excess parents; not additional fits or grid rows.",
        "holdout_success_established":False,
        "notes":["Market forward returns are training supervision only, never forecast inputs.",
            "Past beta ends at the forecast date; all training forward endpoints are within 2020.",
            "Ranking and portfolio metrics still use original stock returns and baseline execution/fees.",
            "Raw/raw-excess controls isolate target changes on the same within-market architecture.",
            "Zero-strength GPU probes exactly match old weights/losses/predictions and are excluded from full results.",
            "This is a local market/stock-component target adaptation, not a FactorVAE reproduction.",
            "Original sampler DropnaLabel and JKP historical release/revision timing remain unverified."]})
    keys = ["IC","ICIR","RankIC","RankICIR","AR","STD","MDD","Sharpe","Sortino","Calmar"]
    pd.DataFrame([{"id":row["id"],"market":row["config"]["market"],
        "target_kind":row["config"]["target_kind"],"strength":row["config"].get("market_residual_strength"),
        "preexisting_control":row["preexisting_complete"],"seconds":row["seconds"],**row["validation"]}
        for row in rows],columns=["id","market","target_kind","strength","preexisting_control","seconds",*keys]).to_csv(
        destination/"market_residual_validation_metrics.csv",index=False)
    shutil.copy2(study/"market_residual_cpu_checks.log",destination/"market_residual_cpu_checks.log")
    print({"completed_configs":len(rows),"new_full_fits":sum(not row["preexisting_complete"] for row in rows),
           "pending":len(pending),"cpu_checks":126,"data_markets":2,"gpu_smokes":6,"test_accessed":False})


if __name__ == "__main__":
    main()
