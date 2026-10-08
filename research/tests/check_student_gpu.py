"""Real two-market Student-t training and unchanged checkpoint recovery harness."""
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from research import ARTIFACTS, ROOT
from research.common import code_fingerprint, digest_file, now, write_json


def main():
    study = ARTIFACTS / "study"
    plan_path = study / "student_smoke_plan.json"
    plan = json.loads(plan_path.read_text())
    records = []
    for item in plan:
        output = Path(item["output"])
        environment = dict(os.environ, KBS_RESEARCH_WORKSPACE_ROOT=str(ROOT),
                           PYTHONPATH=item["bundle"], OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
        with (output / "console.log").open("a") as log:
            subprocess.run([sys.executable, "-u", "-m", "research.train", "--config", item["config"],
                            "--out", str(output)], cwd=item["bundle"], env=environment,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        config = json.loads(Path(item["config"]).read_text())
        result = json.loads((output / "result.json").read_text())
        assert result["status"] == "complete" and result["smoke"] and result["config"] == config
        checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        assert float(checkpoint["model"]["student_df"]) == config["student_df"]
        distribution = json.loads((output / "return_distribution.json").read_text())
        calibration = json.loads((output / "valid/calibration/calibration.json").read_text())
        assert distribution["degrees_of_freedom"] == config["student_df"]
        assert calibration["student_t"]["degrees_of_freedom"] == config["student_df"]
        assert sum(calibration["student_t"]["pit_histogram"]) == calibration["samples"]
        assert len((output / "epochs.jsonl").read_text().splitlines()) == 3
        records.append({**item, "result":result, "distribution":distribution,
                        "calibration":calibration["student_t"],
                        "checkpoint_sha256":digest_file(output / "best.pt")})
        write_json(study / "student_gpu_checks.json", {
            "created_at":now(), "planned":len(plan), "completed":len(records),
            "passed":len(records) == len(plan), "code":code_fingerprint(),
            "scope":"real GPU smoke training/validation; excluded from full trials", "records":records})
        print({"market":config["market"], "student_gpu_smoke":True}, flush=True)
    recovery_path = study / "student_resume_checks.json"
    harness = ROOT / "research/tests/check_mixture_recovery.py"
    subprocess.run([sys.executable, "-u", str(harness), "--plan", str(plan_path),
                    "--record", str(recovery_path)], cwd=Path(__file__).resolve().parents[2], check=True)
    recovery = json.loads(recovery_path.read_text())
    assert recovery["passed"] and len(recovery["records"]) == len(plan)
    recovery.update(cases_completed=len(recovery["records"]),
                    distinct_markets_completed=len({row["market"] for row in recovery["records"]}),
                    harness="research/tests/check_mixture_recovery.py", harness_sha256=digest_file(harness),
                    verifier_sha256=digest_file(Path(__file__)))
    write_json(recovery_path, recovery)
    print({"student_exact_recovery_cases":len(recovery["records"])}, flush=True)


if __name__ == "__main__":
    main()
