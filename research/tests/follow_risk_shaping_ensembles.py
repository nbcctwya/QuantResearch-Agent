"""Follow one verified full-grid process, then compute all fixed seed ensembles."""
import argparse
import json
from pathlib import Path
import time

from research import ARTIFACTS, ROOT
from research.common import code_fingerprint, digest_file, now, write_json
from run_risk_shaping_validation import run_ensembles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-pid", type=int, required=True)
    args = parser.parse_args()
    study = ARTIFACTS/"study"
    design = json.loads((study/"risk_shaping_design.json").read_text())
    assert code_fingerprint() == design["code"]
    assert Path(__import__("research").__file__).parent == Path(design["bundle"])/"research"
    pid = args.worker_pid
    started = time.monotonic()
    activity = study/"risk_shaping_ensemble_follower.json"
    while True:
        if json.loads((ROOT/"research/control.json").read_text()).get("stop"):
            write_json(activity, {"updated_at":now(), "phase":"stopped_by_control", "worker_pid":pid})
            return
        proof = json.loads((study/"risk_shaping_full_trials.json").read_text())
        if proof["complete"]:
            assert proof["planned"] == proof["completed"] == len(design["candidates"]) == 38
            assert proof["code"] == design["code"]
            for row in proof["trials"]:
                folder = Path(row["directory"])
                for name, expected in row["files"].items():
                    assert digest_file(folder/name) == expected
                assert json.loads((folder/"run.json").read_text())["status"] == "complete"
            break
        path = Path(f"/proc/{pid}/cmdline")
        command = path.read_bytes() if path.exists() else b""
        if b"run_risk_shaping_validation.py" not in command or b"--ensembles" in command:
            raise RuntimeError("The specific full-grid worker stopped without completing; inspect its existing proof and log, do not silently restart it")
        if time.monotonic()-started > 4*3600:
            raise TimeoutError("Observation budget exceeded; the live worker is preserved")
        write_json(activity, {"updated_at":now(), "phase":"verified_wait_for_full_grid", "worker_pid":pid,
                             "completed":proof["completed"], "planned":proof["planned"], "code":code_fingerprint()})
        time.sleep(20)
    write_json(activity, {"updated_at":now(), "phase":"actual_seed_ensemble_backtests", "worker_pid":pid, "code":code_fingerprint()})
    run_ensembles(design)
    write_json(activity, {"updated_at":now(), "phase":"complete", "worker_pid":pid, "code":code_fingerprint()})


if __name__ == "__main__":
    main()
