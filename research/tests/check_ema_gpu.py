"""Real EMA GPU trials and independent inspection of interruption checkpoint state."""
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
    plan_path = study / "ema_smoke_plan.json"
    plan = json.loads(plan_path.read_text())
    environment = dict(os.environ, KBS_RESEARCH_WORKSPACE_ROOT=str(ROOT),
                       OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    records = []
    for item in plan:
        output = Path(item["output"])
        with (output / "console.log").open("a") as log:
            subprocess.run([sys.executable, "-u", "-m", "research.train", "--config", item["config"],
                            "--out", str(output)], cwd=item["bundle"], env=environment,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads((output / "result.json").read_text())
        assert result["status"] == "complete" and result["smoke"]
        checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        assert int(checkpoint["ema"]["n_averaged"]) == 24
        progress = [json.loads(line) for line in (output / "epochs.jsonl").read_text().splitlines()]
        assert [row["ema_updates"] for row in progress] == [8, 16, 24]
        best = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
        if best["epoch"] == checkpoint["epoch"]:
            for name, value in best["model"].items():
                torch.testing.assert_close(value, checkpoint["ema"]["module." + name], atol=0, rtol=0)
        records.append({**item, "result": result, "ema_updates": 24,
                        "averaging": json.loads((output / "weight_averaging.json").read_text()),
                        "checkpoint_sha256": digest_file(output / "best.pt")})
        write_json(study / "ema_gpu_checks.json", {"created_at": now(), "planned": len(plan),
                   "completed": len(records), "passed": len(records) == len(plan),
                   "scope": "GPU smoke training/validation only; excluded from full trials", "records": records,
                   "code": code_fingerprint()})
        print({"case": item["case"], "ema_gpu_smoke": True}, flush=True)
    recovery_path = study / "ema_resume_checks.json"
    subprocess.run([sys.executable, "-u", "-m", "research.tests.check_mixture_recovery",
                    "--plan", str(plan_path), "--record", str(recovery_path)], cwd=ROOT,
                   env=environment, check=True)
    recovery = json.loads(recovery_path.read_text())
    for record in recovery["records"]:
        original = Path(record["checkpoint_before_best_comparison"]).parent
        recovered = Path(record["recovered_directory"])
        expected = torch.load(original / "last.pt", map_location="cpu", weights_only=False)
        actual = torch.load(recovered / "last.pt", map_location="cpu", weights_only=False)
        assert set(expected["ema"]) == set(actual["ema"])
        for name, value in expected["ema"].items():
            torch.testing.assert_close(value, actual["ema"][name], atol=0, rtol=0)
        record["exact_ema_parameters_and_update_counter"] = True
        record["ema_updates"] = int(actual["ema"]["n_averaged"])
    recovery.update(cases_completed=len(recovery["records"]),
                    distinct_markets_completed=len({r["market"] for r in recovery["records"]}),
                    harness="research/tests/check_mixture_recovery.py",
                    harness_sha256=digest_file(ROOT / "research/tests/check_mixture_recovery.py"),
                    ema_state_verifier="research/tests/check_ema_gpu.py",
                    ema_state_verifier_sha256=digest_file(Path(__file__)))
    write_json(recovery_path, recovery)
    print({"exact_ema_recovery_cases": len(recovery["records"])}, flush=True)


if __name__ == "__main__":
    main()
