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
    path = study/"robust_loss_seed_design.json"
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
            write_json(study/"robust_loss_seed_follower.json",{"updated_at":now(),
                "phase":"stopped_by_control","test_accessed":False})
            return
        state = json.loads((study/"state.json").read_text())
        failures = sorted(set(state["failed"]) & {item["id"] for item in plan["candidates"]})
        if failures:
            raise RuntimeError("Fixed volatility-target seed fits failed: "+str(failures))
        ready = []
        for item in plan["candidates"]:
            folder = ARTIFACTS/"trials"/item["id"]
            if (folder/"result.json").exists() and json.loads((folder/"run.json").read_text())["status"] == "complete":
                ready.append(item["id"])
        write_json(study/"robust_loss_seed_follower.json",{"updated_at":now(),"phase":"waiting_for_fixed_source_fits",
            "pid":os.getpid(),"ready":len(ready),"planned":36,"plan_sha256":plan_hash,"code":plan["code"],"test_accessed":False})
        if len(ready) == 36:
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
    for control in plan["reference_control_sources"]:
        assert artifact_record(control["config"]) == control
    files = [ARTIFACTS/"trials"/source["id"]/name for source in sources
             for name in ["config.json","run.json","result.json","best.pt","last.pt","valid/predictions.pkl","robust_loss.json","target_transform.json"]]
    before = source_state(files)
    records = []
    for candidate in plan["methods"]:
        proofs = [json.loads((ARTIFACTS/"trials"/config_id({**candidate,"seed":seed})/"robust_loss.json").read_text())
                  for seed in plan["validation_seeds"]]
        assert all(proof==proofs[0] for proof in proofs)
        assert proofs[0]["huber_delta"]==candidate["huber_delta"]
        transforms=[json.loads((ARTIFACTS/"trials"/config_id({**candidate,"seed":seed})/"target_transform.json").read_text())
                    for seed in plan["validation_seeds"]]
        assert all(transform==transforms[0] for transform in transforms)
        record = ensemble_validation(candidate,plan["validation_seeds"],plan)
        verify_confirmation_record(record)
        records.append({"name":"robust_loss_"+config_id(candidate)+"_seeds012",**record})
        write_json(study/"robust_loss_seed_validation_checks.json",{"updated_at":now(),
            "scope":plan["scope"],"code":plan["code"],"plan_sha256":plan_hash,"completed":len(records),
            "planned":12,"records":records,"source_files_unchanged":False})
        print({"method":config_id(candidate),"market":candidate["market"],"ensembles_completed":len(records)},flush=True)
    assert before == source_state(files)
    for control in plan["reference_control_sources"]:
        assert artifact_record(control["config"]) == control
    report = {"updated_at":now(),"scope":plan["scope"],"code":plan["code"],"plan_sha256":plan_hash,
        "completed":12,"planned":12,"records":records,"source_files_unchanged":True,
        "checked_source_files":before,"harness_sha256":digest_file(Path(__file__)),"test_accessed":False}
    write_json(study/"robust_loss_seed_validation_checks.json",report)
    write_json(study/"robust_loss_seed_follower.json",{"updated_at":now(),"phase":"complete",
        "pid":os.getpid(),"ready":36,"planned":36,"ensembles_completed":12,"plan_sha256":plan_hash,"test_accessed":False})
    print({"actual_seed_ensembles":12,"source_files_unchanged":len(before),"test_accessed":False},flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_json(ARTIFACTS/"study/robust_loss_seed_follower_failure.json",{
            "updated_at":now(),"error_type":type(error).__name__,"error":str(error),"test_accessed":False})
        raise
