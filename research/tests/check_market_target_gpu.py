"""Two-market CUDA probes, exact disabled controls, and interrupted recovery."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import torch

from research import ARTIFACTS
from research.common import code_fingerprint, digest_file, now, write_json


def main():
    study = ARTIFACTS/"study"
    design = json.loads((study/"market_residual_design.json").read_text())
    assert code_fingerprint() == design["code"]
    assert Path(__import__("research").__file__).parent == Path(design["bundle"])/"research"
    completed = []
    for item in design["smokes"]:
        folder, bundle = Path(item["output"]), Path(item["bundle"])
        expected_code = json.loads((bundle/"manifest.json").read_text())["files"]
        config = json.loads(Path(item["config"]).read_text())
        environment = dict(os.environ, KBS_RESEARCH_WORKSPACE_ROOT=str(ARTIFACTS.parent.parent),
            PYTHONPATH=str(bundle), OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
        if not (folder/"result.json").exists():
            with (folder/"console.log").open("a") as stream:
                subprocess.run([sys.executable,"-u","-m","research.train","--config",item["config"],
                    "--out",str(folder)],cwd=bundle,env=environment,
                    stdout=stream,stderr=subprocess.STDOUT,check=True)
        result=json.loads((folder/"result.json").read_text())
        run=json.loads((folder/"run.json").read_text())
        assert result["config"] == config and result["status"] == "complete" and result["smoke"]
        assert run["code"] == expected_code and run["source_package"] == str(bundle/"research")
        checkpoint=torch.load(folder/"last.pt",map_location="cpu",weights_only=False)
        if config["target_kind"] == "market_residual_standardized":
            proof=json.loads((folder/"market_residual.json").read_text())
            assert checkpoint["training_target_signature"] == proof["target_signature"]
            assert not proof["forecast_inputs_changed"] and not proof["test_targets_used"]
        completed.append({**item,"result_sha256":digest_file(folder/"result.json"),
            "run_sha256":digest_file(folder/"run.json"),"source_code":expected_code})
        write_json(study/"market_residual_gpu_progress.json",{"updated_at":now(),
            "smokes_completed":len(completed),"planned_smokes":6,"records":completed})
        print({"market":item["market"],"case":item["case"],"smoke_complete":True},flush=True)
    controls=[]
    for pair in json.loads(Path(design["disabled_plan"]).read_text()):
        old,new=Path(pair["old"]["output"]),Path(pair["new"]["output"])
        for name in ["last.pt","best.pt"]:
            expected=torch.load(old/name,map_location="cpu",weights_only=False)
            actual=torch.load(new/name,map_location="cpu",weights_only=False)
            assert expected["epoch"] == actual["epoch"]
            for key,value in expected["model"].items():
                torch.testing.assert_close(value,actual["model"][key],rtol=0,atol=0)
        old_epochs=[json.loads(line) for line in (old/"epochs.jsonl").read_text().splitlines()]
        new_epochs=[json.loads(line) for line in (new/"epochs.jsonl").read_text().splitlines()]
        assert len(old_epochs) == len(new_epochs) == 3
        for left,right in zip(old_epochs,new_epochs):
            for key in ["epoch","train_loss","valid_selection_score","best_selection_score","valid","optimizer_steps"]:
                assert left[key] == right[key], (pair["market"],key,left[key],right[key])
        pd.testing.assert_frame_equal(pd.read_pickle(old/"valid/predictions.pkl"),
            pd.read_pickle(new/"valid/predictions.pkl"),check_exact=True)
        controls.append({"market":pair["market"],"legacy_source":pair["old"],"new_source":pair["new"],
            "exact_last_and_best_model_parameters":True,"exact_losses":True,
            "exact_original_raw_label_validation_predictions":True})
    environment=dict(os.environ, PYTHONPATH=design["bundle"],
        KBS_RESEARCH_WORKSPACE_ROOT=str(ARTIFACTS.parent.parent),OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2")
    helper=ARTIFACTS.parent/"tests/check_mixture_recovery.py"
    with (study/"market_residual_recovery_checks.log").open("a") as stream:
        subprocess.run([sys.executable,"-u",str(helper),"--plan",design["recovery_plan"],
            "--record",str(study/"market_residual_recovery_checks.json")],cwd=design["bundle"],
            env=environment,stdout=stream,stderr=subprocess.STDOUT,check=True)
    recovery=json.loads((study/"market_residual_recovery_checks.json").read_text())
    assert recovery["passed"] and recovery["markets_completed"] == 2
    write_json(study/"market_residual_gpu_checks.json",{"created_at":now(),"passed":True,
        "code":code_fingerprint(),"scope":"six GPU smoke runs and two exact interruption/recovery probes; excluded from full validation counts; no test",
        "harness_sha256":digest_file(Path(__file__)),"recovery_harness_sha256":digest_file(helper),
        "smokes":completed,"zero_strength_controls":controls,"recovery":recovery})
    print({"gpu_smokes":6,"exact_disabled_controls":2,"exact_interrupt_recovery":2,"passed":True},flush=True)


if __name__ == "__main__":
    main()
