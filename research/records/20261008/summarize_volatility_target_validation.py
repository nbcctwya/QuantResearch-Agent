"""Export only genuine full normalized-target runs and their immutable evidence."""
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd
import torch

from research import ARTIFACTS, ROOT
from research.common import config_id, digest_file, now, write_json
from research.selection import artifact_record, verify_confirmation_record, verify_plan


def main():
    study = ARTIFACTS/"study"
    destination = ROOT/"research/records/20261008"
    design = json.loads((study/"volatility_target_design.json").read_text())
    bundle = Path(design["bundle"])
    assert digest_file(bundle/"manifest.json") == design["code_manifest_sha256"]
    assert all(digest_file(bundle/name)==value for name,value in design["code"].items())
    checks = {name:json.loads((study/(name+".json")).read_text())
        for name in ["volatility_target_data_checks","volatility_target_gpu_checks"]}
    assert all(check["passed"] and check["code"]==design["code"] for check in checks.values())
    source = study/"volatility_target_full_trials.json"
    progress = json.loads(source.read_text()) if source.exists() else {"completed":0,"planned":8,"trials":[],"complete":False}
    rows = []
    for item in progress["trials"]:
        assert config_id(item["config"]) == item["id"]
        original = Path(item["directory"])
        folder = original if original.exists() else ARTIFACTS/"trials"/item["id"]
        assert all(digest_file(folder/name)==value for name,value in item["files"].items())
        result=json.loads((folder/"result.json").read_text());run=json.loads((folder/"run.json").read_text())
        assert result["config"]==item["config"] and run["status"]==result["status"]=="complete" and not result.get("smoke")
        assert run["code"]==design["code"] and run["source_package"]==str(bundle/"research")
        proof=json.loads((folder/"volatility_target.json").read_text())
        assert proof["spec"]["base"]==item["config"]["volatility_target_base"]
        assert proof["spec"]["mode"]==item["config"]["volatility_target_mode"]
        assert proof["spec"]["power"]==item["config"]["volatility_target_power"]
        signature={name:proof[name] for name in ["spec","raw_return_sha256","target_sha256","scale",
            "forecast_index_sha256","volatility_cache_manifest_sha256","volatility_statistics","zero_power_base_signature"]}
        assert hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()==proof["target_signature"]
        checkpoint=torch.load(folder/"last.pt",map_location="cpu",weights_only=False)
        assert checkpoint["training_target_signature"]==proof["target_signature"]
        assert not proof["forecast_inputs_changed"] and not proof["test_targets_used"]
        data=proof["volatility_data"];cache=Path(data["path"])
        assert digest_file(cache/"manifest.json")==data["manifest_sha256"]
        assert all(digest_file(cache/name)==value for name,value in data["files"].items())
        assert pd.Timestamp(data["stock_price_end"])<=pd.Timestamp("2020-12-31")
        parent=data["parent_market_data"];parent_cache=Path(parent["path"])
        assert digest_file(parent_cache/"manifest.json")==parent["manifest_sha256"]
        assert all(digest_file(parent_cache/name)==value for name,value in parent["files"].items())
        rows.append({**item,"current_directory":str(folder),"run":run,"validation":result["validation"],
            "resources":result.get("resources"),"seconds":result["seconds"],"target_proof":proof})
    assert len(rows)==progress["completed"]
    controls=[]
    for item in design["existing_controls"]:
        folder=Path(item["directory"]);result=json.loads((folder/"result.json").read_text())
        assert result["config"]==item["config"] and result["status"]=="complete" and not result.get("smoke")
        controls.append({**item,"validation":result["validation"],
            "files":{name:digest_file(folder/name) for name in ["run.json","result.json","best.pt","valid/predictions.pkl"]}})
    saved=destination/"code_snapshots"/bundle.name
    if not saved.exists():shutil.copytree(bundle,saved,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    installed=study/"volatility_target_install.json"
    events=study/"volatility_target_export_events.json"
    imported=study/"volatility_target_import.json"
    seed_study=None
    seed_path=study/"volatility_target_seed_design.json"
    if seed_path.exists():
        seed_design=json.loads(seed_path.read_text())
        assert seed_design["code"]==design["code"]
        verify_plan(seed_design)
        for control in seed_design["reference_control_sources"]:
            assert artifact_record(control["config"])==control
        ready,pending=[],[]
        for item in seed_design["candidates"]:
            folder=ARTIFACTS/"trials"/item["id"]
            if not (folder/"result.json").exists() or json.loads((folder/"run.json").read_text())["status"]!="complete":
                pending.append(item["id"])
                continue
            record=artifact_record(item["config"])
            assert record["run"]["code"]==design["code"] and record["run"]["source_package"]==str(bundle/"research")
            ready.append(record)
        seed_checks_path=study/"volatility_target_seed_validation_checks.json"
        seed_checks=json.loads(seed_checks_path.read_text()) if seed_checks_path.exists() else None
        queue=study/"volatility_target_seed_queue.json"
        seed_study={"design":seed_design,"design_sha256":digest_file(seed_path),
            "completed_source_fits":len(ready),"planned_source_fits":24,"ready":ready,"pending":pending,
            "queue":json.loads(queue.read_text()) if queue.exists() else None,
            "follower":json.loads((study/"volatility_target_seed_follower.json").read_text()),
            "jobs":json.loads((study/"volatility_target_seed_jobs.json").read_text()),
            "actual_seed_ensembles":seed_checks}
        if seed_checks and seed_checks["completed"]==seed_checks["planned"]==8 and seed_checks["source_files_unchanged"]:
            existing_path=destination/"seed_validation_checks.json"
            existing=json.loads(existing_path.read_text());names={row["name"]:row for row in existing["records"]}
            for record in seed_checks["records"]:
                verify_confirmation_record(record)
                summary={**record,"market":record["config"]["market"],"candidate":record["config"],
                    "ensemble":record["validation"],"output":record["directory"],"code":seed_checks["code"],
                    "scope":"actual fixed three-seed volatility-target validation; unchanged baseline evaluation; no test"}
                if record["name"] in names:assert names[record["name"]]==summary
                else:existing["records"].append(summary)
            existing["updated_at"]=now();write_json(existing_path,existing)
    write_json(destination/"volatility_target_validation_checks.json",{"created_at":now(),
        "scope":"fixed eight actual neural target fits and four existing controls; original baseline validation evaluation; no test",
        "design":design,"checks":checks,"cpu_tests":136,"cpu_log_sha256":digest_file(study/"volatility_target_cpu_checks.log"),
        "planned_new_full_fits":8,"completed_new_full_fits":len(rows),"experiments":rows,"existing_controls":controls,
        "pending":[config_id(config) for config in design["candidates"] if config_id(config) not in {row["id"] for row in rows}],
        "install":json.loads(installed.read_text()) if installed.exists() else None,
        "jobs":json.loads((study/"volatility_target_jobs.json").read_text()),
        "reporting_harness_events":json.loads(events.read_text()) if events.exists() else [],
        "import":json.loads(imported.read_text()) if imported.exists() else None,"seed_study":seed_study,
        "notes":["All transformed targets and volatility floor/median are fitted on selected training rows only.",
            "Zero exponent exactly matches original targets, trained weights, losses and predictions; smoke probes excluded.",
            "Old market-target implementation and its source caches stay byte-identical.",
            "Only past history_prices contributes to risk estimates; future market returns are supervision for the residual base only.",
            "Label adaptation does not introduce volatility sizing, leverage, new fees or new execution rules.",
            "Current outputs are genuine independent full CUDA fits; temporary folders are imported only after fixed-grid completion.",
            "Inherited sampler DropnaLabel and JKP historical release/revision timing remain unverified."],
        "holdout_success_established":False})
    keys=["IC","ICIR","RankIC","RankICIR","AR","STD","MDD","Sharpe","Sortino","Calmar"]
    pd.DataFrame([{"id":row["id"],"market":row["config"]["market"],
        "base":row["config"]["volatility_target_base"],"risk_mode":row["config"]["volatility_target_mode"],
        "power":row["config"]["volatility_target_power"],"seconds":row["seconds"],**row["validation"]}
        for row in rows],columns=["id","market","base","risk_mode","power","seconds",*keys]).to_csv(
        destination/"volatility_target_validation_metrics.csv",index=False)
    if seed_study and seed_study["actual_seed_ensembles"]:
        pd.DataFrame([{"name":row["name"],"market":row["config"]["market"],
            "base":row["config"]["volatility_target_base"],"risk_mode":row["config"]["volatility_target_mode"],
            "power":row["config"]["volatility_target_power"],"seeds":"0,1,2",**row["validation"]}
            for row in seed_study["actual_seed_ensembles"]["records"]],
            columns=["name","market","base","risk_mode","power","seeds",*keys]).to_csv(
                destination/"volatility_target_seed_validation_metrics.csv",index=False)
    shutil.copy2(study/"volatility_target_cpu_checks.log",destination/"volatility_target_cpu_checks.log")
    print({"complete_full_new_fits":len(rows),"planned":8,"existing_controls":4,"cpu_checks":136,"gpu_smokes":8,"test_accessed":False})


if __name__=="__main__":
    main()
