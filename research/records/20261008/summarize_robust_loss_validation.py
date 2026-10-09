"""Archive checked staged robust loss and its actual full-model progress."""
import json
from pathlib import Path
import shutil

import pandas as pd

from research import ARTIFACTS,ROOT
from research.common import code_fingerprint,config_id,digest_file,now,write_json


def main():
    study=ARTIFACTS/"study";destination=ROOT/"research/records/20261008"
    design=json.loads((study/"robust_loss_design.json").read_text());bundle=Path(design["bundle"])
    assert code_fingerprint()==design["code"]
    assert Path(__import__("research").__file__).parent==bundle/"research"
    assert digest_file(bundle/"manifest.json")==design["code_manifest_sha256"]
    assert all(digest_file(bundle/name)==value for name,value in design["code"].items())
    stage=json.loads((study/"robust_loss_stage.json").read_text())
    assert stage["cpu_tests_passed"]==146 and digest_file(study/"robust_loss_cpu_checks.log")==stage["cpu_log_sha256"]
    checks={name:json.loads((study/(name+".json")).read_text()) for name in ["robust_loss_data_checks","robust_loss_gpu_checks"]}
    assert all(value["passed"] and value["code"]==design["code"] for value in checks.values())
    progress_path=study/"robust_loss_full_trials.json"
    progress=json.loads(progress_path.read_text()) if progress_path.exists() else {"completed":0,"planned":12,"trials":[],"complete":False}
    rows=[]
    for item in progress["trials"]:
        assert item["id"]==config_id(item["config"])
        original=Path(item["directory"]);folder=original if original.exists() else ARTIFACTS/"trials"/item["id"]
        assert all(digest_file(folder/name)==value for name,value in item["files"].items())
        result=json.loads((folder/"result.json").read_text());run=json.loads((folder/"run.json").read_text())
        assert result["config"]==item["config"] and run["status"]==result["status"]=="complete" and not result.get("smoke")
        assert run["code"]==design["code"] and run["source_package"]==str(bundle/"research")
        proof=json.loads((folder/"robust_loss.json").read_text())
        assert proof["huber_delta"]==item["config"]["huber_delta"] and proof["mixed_weights"]=={"regression":.5,"correlation":.5}
        transform=json.loads((folder/"target_transform.json").read_text())
        control=next(row for row in design["existing_controls"] if row["config"]["market"]==item["config"]["market"]
            and row["config"]["target_kind"]==item["config"]["target_kind"])
        assert transform==json.loads((Path(control["directory"])/"target_transform.json").read_text())
        rows.append({**item,"current_directory":str(folder),"run":run,"validation":result["validation"],
            "seconds":result["seconds"],"resources":result.get("resources"),"robust_loss":proof,"target_transform":transform})
    assert len(rows)==progress["completed"]
    controls=[]
    for row in design["existing_controls"]:
        folder=Path(row["directory"]);assert all(digest_file(folder/name)==value for name,value in row["files"].items())
        result=json.loads((folder/"result.json").read_text());assert result["config"]==row["config"] and result["status"]=="complete" and not result.get("smoke")
        controls.append({**row,"validation":result["validation"]})
    saved=destination/"code_snapshots"/bundle.name
    if not saved.exists():shutil.copytree(bundle,saved,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    failures=["robust_loss_cpu_checks_initial_import.log","robust_loss_gpu_checks_initial_source_path.log","robust_loss_data_checks_initial_source_path.log"]
    for name in ["robust_loss_cpu_checks.log",*failures]:shutil.copy2(study/name,destination/"staged_robust_loss"/name)
    seed_path=study/"robust_loss_seed_design.json"
    seed_design=json.loads(seed_path.read_text()) if seed_path.exists() else None
    if seed_design:
        from research.selection import verify_plan,artifact_record
        verify_plan(seed_design)
        for source in seed_design["reference_control_sources"]:assert artifact_record(source["config"])==source
    write_json(destination/"robust_loss_validation_checks.json",{"created_at":now(),
        "scope":"staged checked robust-loss source and actual independent full neural validation outputs; no test",
        "design":design,"stage":stage,"checks":checks,"planned_new_full_fits":12,"completed_new_full_fits":len(rows),
        "experiments":rows,"existing_controls":controls,"pending":[config_id(config) for config in design["candidates"] if config_id(config) not in {row["id"] for row in rows}],
        "jobs":json.loads((study/"robust_loss_jobs.json").read_text()),
        "seed_design":seed_design,"additional_seed_fits_planned":24 if seed_design else None,
        "additional_seed_fits_queued":(study/"robust_loss_seed_queue.json").exists(),
        "initial_check_failures":{name:digest_file(study/name) for name in failures},
        "failure_note":"Initial unittest command preferred the live workspace, then initial data/GPU harnesses preferred the sibling staging package and refused the immutable-package guard. Corrected execution locations and independent harness directory; checked core source and tolerances unchanged.",
        "live_source_installed":(study/"robust_loss_install.json").exists(),
        "install_and_import_gate":"Finish all24 volatility-target source fits and install checked robust code before importing these12 outputs; avoid a coordinator that ignores robust config keys.",
        "holdout_success_established":False})
    keys=["IC","ICIR","RankIC","RankICIR","AR","STD","MDD","Sharpe","Sortino","Calmar"]
    pd.DataFrame([{"id":row["id"],"market":row["config"]["market"],"target_kind":row["config"]["target_kind"],
        "huber_delta":row["config"]["huber_delta"],"seconds":row["seconds"],**row["validation"]} for row in rows],
        columns=["id","market","target_kind","huber_delta","seconds",*keys]).to_csv(destination/"robust_loss_validation_metrics.csv",index=False)
    print({"completed_new_full_fits":len(rows),"planned":12,"cpu_tests":146,"actual_full_data_markets":2,"gpu_smokes":6,"exact_disabled_controls":4,"exact_recovery_cases":2,"live_core_installed":False})


if __name__=="__main__":main()
