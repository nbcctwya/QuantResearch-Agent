"""Wait for all fixed target fits; reuse the checked exact validation ensemble path."""
import json
import os
from pathlib import Path
import time

from research import ARTIFACTS, ROOT
from research.common import code_fingerprint, config_id, digest_file, now, write_json
from research.selection import artifact_record, ensemble_validation, verify_confirmation_record, verify_plan


def source_state(paths):
    return {str(path):{"sha256":digest_file(path),"mtime_ns":path.stat().st_mtime_ns} for path in paths}


def main():
    study = ARTIFACTS/"study"
    path = study/"market_residual_seed_design.json"
    plan = json.loads(path.read_text())
    plan_hash = digest_file(path)
    assert code_fingerprint() == plan["code"]
    assert Path(__import__("research").__file__).parent == Path(plan["code_bundle"])/"research"
    deadline = time.monotonic()+6*3600
    while True:
        assert digest_file(path) == plan_hash
        verify_plan(plan)
        control = json.loads((ROOT/"research/control.json").read_text())
        if control.get("stop"):
            write_json(study/"market_residual_seed_follower.json",{"updated_at":now(),
                "phase":"stopped_by_control","test_accessed":False})
            return
        state = json.loads((study/"state.json").read_text())
        failures = sorted(set(state["failed"]) & {item["id"] for item in plan["candidates"]})
        if failures:
            raise RuntimeError("Fixed target seed fits failed: "+str(failures))
        ready = []
        for item in plan["candidates"]:
            folder = ARTIFACTS/"trials"/item["id"]
            if (folder/"result.json").exists() and json.loads((folder/"run.json").read_text())["status"] == "complete":
                ready.append(item["id"])
        write_json(study/"market_residual_seed_follower.json",{"updated_at":now(),"phase":"waiting_for_fixed_source_fits",
            "pid":os.getpid(),"ready":len(ready),"planned":24,"plan_sha256":plan_hash,"code":plan["code"],"test_accessed":False})
        if len(ready) == 24:
            break
        if time.monotonic() > deadline:
            raise RuntimeError("Fixed source fits have not finished within follower deadline; do not restart live training")
        time.sleep(20)
    sources = [artifact_record(item["config"]) for item in plan["candidates"]]
    for source in sources:
        if source["id"] in plan["preexisting"]:
            assert source == plan["preexisting"][source["id"]]
        else:
            assert source["run"]["code"] == plan["code"]
            assert source["run"]["source_package"] == str(Path(plan["code_bundle"])/"research")
    files = [ARTIFACTS/"trials"/source["id"]/name for source in sources
             for name in ["config.json","run.json","result.json","best.pt","valid/predictions.pkl"]]
    before = source_state(files)
    records = []
    for candidate in plan["methods"]:
        if candidate["target_kind"] == "market_residual_standardized":
            proofs = [json.loads((ARTIFACTS/"trials"/config_id({**candidate,"seed":seed})/"market_residual.json").read_text())
                      for seed in plan["validation_seeds"]]
            assert len({proof["target_signature"] for proof in proofs}) == 1
            for proof in proofs:
                assert not proof["forecast_inputs_changed"] and not proof["test_targets_used"]
                data = proof["market_data"]
                assert digest_file(Path(data["path"])/"manifest.json") == data["manifest_sha256"]
                assert all(digest_file(Path(data["path"])/name) == value for name,value in data["files"].items())
        record = ensemble_validation(candidate,plan["validation_seeds"],plan)
        verify_confirmation_record(record)
        records.append({"name":"market_residual_"+config_id(candidate)+"_seeds012",**record})
        write_json(study/"market_residual_seed_validation_checks.json",{"updated_at":now(),
            "scope":plan["scope"],"code":plan["code"],"plan_sha256":plan_hash,"completed":len(records),
            "planned":8,"records":records,"source_files_unchanged":False})
        print({"method":config_id(candidate),"market":candidate["market"],"ensembles_completed":len(records)},flush=True)
    assert before == source_state(files)
    report = {"updated_at":now(),"scope":plan["scope"],"code":plan["code"],"plan_sha256":plan_hash,
        "completed":8,"planned":8,"records":records,"source_files_unchanged":True,
        "checked_source_files":before,"harness_sha256":digest_file(Path(__file__)),"test_accessed":False}
    write_json(study/"market_residual_seed_validation_checks.json",report)
    write_json(study/"market_residual_seed_follower.json",{"updated_at":now(),"phase":"complete",
        "pid":os.getpid(),"ready":24,"planned":24,"ensembles_completed":8,"plan_sha256":plan_hash,"test_accessed":False})
    print({"actual_seed_ensembles":8,"source_files_unchanged":len(before),"test_accessed":False},flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_json(ARTIFACTS/"study/market_residual_seed_follower_failure.json",{
            "updated_at":now(),"error_type":type(error).__name__,"error":str(error),"test_accessed":False})
        raise
