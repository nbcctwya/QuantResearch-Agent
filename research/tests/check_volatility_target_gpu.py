"""CUDA target probes with exact old-base controls and saved-epoch recovery."""
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
    design = json.loads((study/"volatility_target_design.json").read_text())
    assert code_fingerprint() == design["code"]
    assert Path(__import__("research").__file__).parent == Path(design["bundle"])/"research"
    environment = dict(os.environ,PYTHONPATH=design["bundle"],
        KBS_RESEARCH_WORKSPACE_ROOT=str(ARTIFACTS.parent.parent),OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2")
    completed = []
    for item in design["smokes"]:
        folder = Path(item["output"])
        config = json.loads(Path(item["config"]).read_text())
        if not (folder/"result.json").exists():
            with (folder/"console.log").open("a") as stream:
                subprocess.run([sys.executable,"-u","-m","research.train","--config",item["config"],
                    "--out",str(folder)],cwd=item["bundle"],env=environment,
                    stdout=stream,stderr=subprocess.STDOUT,check=True)
        result = json.loads((folder/"result.json").read_text())
        run = json.loads((folder/"run.json").read_text())
        proof = json.loads((folder/"volatility_target.json").read_text())
        assert result["config"] == config and result["status"] == "complete" and result["smoke"]
        assert run["code"] == design["code"] and run["source_package"] == str(Path(design["bundle"])/"research")
        checkpoint = torch.load(folder/"last.pt",map_location="cpu",weights_only=False)
        assert checkpoint["training_target_signature"] == proof["target_signature"]
        assert not proof["forecast_inputs_changed"] and not proof["test_targets_used"]
        completed.append({**item,"result_sha256":digest_file(folder/"result.json"),
            "run_sha256":digest_file(folder/"run.json"),"target_proof_sha256":digest_file(folder/"volatility_target.json")})
        write_json(study/"volatility_target_gpu_progress.json",{"updated_at":now(),"completed":len(completed),
            "planned":8,"records":completed})
        print({"market":item["market"],"case":item["case"],"complete":True},flush=True)
    controls = []
    for pair in json.loads(Path(design["disabled_plan"]).read_text()):
        old,new = Path(pair["old"]["output"]),Path(pair["new"]["output"])
        for name in ["last.pt","best.pt"]:
            expected = torch.load(old/name,map_location="cpu",weights_only=False)
            actual = torch.load(new/name,map_location="cpu",weights_only=False)
            assert expected["epoch"] == actual["epoch"]
            for key,value in expected["model"].items():
                torch.testing.assert_close(value,actual["model"][key],rtol=0,atol=0)
        old_epochs = [json.loads(line) for line in (old/"epochs.jsonl").read_text().splitlines()]
        new_epochs = [json.loads(line) for line in (new/"epochs.jsonl").read_text().splitlines()]
        assert len(old_epochs) == len(new_epochs) == 3
        for left,right in zip(old_epochs,new_epochs):
            for key in ["epoch","train_loss","valid_selection_score","best_selection_score","valid","optimizer_steps"]:
                assert left[key] == right[key],(pair["market"],pair["base"],key,left[key],right[key])
        pd.testing.assert_frame_equal(pd.read_pickle(old/"valid/predictions.pkl"),
            pd.read_pickle(new/"valid/predictions.pkl"),check_exact=True)
        controls.append({"market":pair["market"],"base":pair["base"],"legacy_reference":pair["old"],
            "new_zero_probe":pair["new"],"exact_model_parameters":True,"exact_losses":True,
            "exact_original_raw_label_predictions_and_dtype":True})
    recovery_helper = ARTIFACTS.parent/"tests/check_mixture_recovery.py"
    with (study/"volatility_target_recovery_checks.log").open("a") as stream:
        subprocess.run([sys.executable,"-u",str(recovery_helper),"--plan",design["recovery_plan"],
            "--record",str(study/"volatility_target_recovery_checks.json")],cwd=design["bundle"],env=environment,
            stdout=stream,stderr=subprocess.STDOUT,check=True)
    recovery = json.loads((study/"volatility_target_recovery_checks.json").read_text())
    assert recovery["passed"] and recovery["markets_completed"] == 4
    write_json(study/"volatility_target_gpu_checks.json",{"created_at":now(),"passed":True,
        "code":code_fingerprint(),"scope":"eight smoke runs, four exact disabled controls and four exact recovery probes; excluded from full results; no test",
        "smokes":completed,"zero_power_controls":controls,"recovery":recovery,
        "harness_sha256":digest_file(Path(__file__)),"recovery_harness_sha256":digest_file(recovery_helper)})
    print({"gpu_smokes":8,"exact_disabled_controls":4,"exact_saved_epoch_recovery_cases":4,"passed":True},flush=True)


if __name__ == "__main__":
    main()
