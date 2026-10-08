"""Export actual selection/recovery evidence without creating an official test lock."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

from research import ARTIFACTS, ROOT
from research.common import code_fingerprint, digest_file, now, write_json


def summarize():
    destination = ROOT/"research/records/20261008"
    study = ARTIFACTS/"study"
    proof = json.loads((destination/"validation_selection_checks.json").read_text())
    cpu = json.loads((study/"selection_cpu_checks.json").read_text())
    install = json.loads((study/"selection_install.json").read_text())
    assert cpu["passed"] and cpu["tests"] == 108
    assert cpu["code"] == proof["code"] == install["code"] == code_fingerprint()
    assert cpu["log_sha256"] == digest_file(study/"selection_cpu_checks.log")
    assert cpu["test_source_sha256"] == digest_file(ROOT/"research/tests/test_selection.py")
    assert proof["no_refits"] and proof["cached_restart_extra_backtests"] == 0
    assert all(row["exact_predictions_and_dtype"] and row["max_metric_difference"] <= 1e-12 for row in proof["checks"])
    assert not (study/"selection_lock.json").exists()
    assert not (study/"validation_confirmation_plan.json").exists()
    assert not list((ARTIFACTS/"holdout").glob("*/test_run.json"))
    bundle = Path(install["bundle"])
    manifest = json.loads((bundle/"manifest.json").read_text())
    assert manifest["files"] == proof["code"]
    assert all(digest_file(bundle/name) == expected for name, expected in manifest["files"].items())
    saved = destination/"code_snapshots"/bundle.name
    if not saved.exists():
        shutil.copytree(bundle, saved, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    initial = []
    original_completed_backtests = []
    for filename, reason in [
        ("validation_selection_checks_initial.log", "The preview harness omitted IDs normally supplied by production read_results; corrected the harness without changing the selector."),
        ("validation_selection_checks_reference_path_initial.log", "All six actual backtests completed; the harness looked for three older reference ensembles in validation_checks instead of study/seed_validation. Corrected the reference path and reused all six completed backtests.")]:
        path = study/filename
        lines = path.read_text().splitlines()
        for line in lines:
            if line.startswith('{"event": "backtest_complete"'):
                original_completed_backtests.append(json.loads(line))
        initial.append({"file":str(path), "sha256":digest_file(path), "reason":reason,
                        "final_error":lines[-1], "classified_as_model_trial_failure":False})
    assert len(original_completed_backtests) == 6
    write_json(destination/"validation_selection_implementation_checks.json", {
        "created_at":now(), "scope":"validation-only implementation and restart audit; six existing ensemble methods recomputed, not six new model fits or extra unique seed studies",
        "cpu":cpu, "real_preview_report":"validation_selection_checks.json", "install":install,
        "initial_harness_checks":initial, "original_completed_backtests":original_completed_backtests,
        "restart_extra_backtests":0, "new_model_holdout_records":0, "official_selection_plan_exists":False,
        "notes":["Validation shortlist size 12 per market, three validation seeds, three promotions per market and five final seeds are fixed in control before final search selection.",
                 "Search proposals and single-checkpoint selection are unchanged; only final candidate confirmation changes.",
                 "All ten actual validation-ensemble metrics use baseline computation; metric averaging is never substituted for backtesting.",
                 "Balanced validation utility does not use holdout baseline thresholds; ties and metric directions are explicit.",
                 "Preview winners are representative pipeline checks, not official candidates or a finalized research result.",
                 "Protocol evidence hashes baseline evaluator source files and four dataset manifests, not the full cached data arrays.",
                 "Preexisting fits retain their original training source records; the new evaluation package is not attributed retroactively.",
                 "Older lock formats remain readable for compatibility; every new coordinator lock includes the complete validation and five-seed evidence requirements."]})
    shutil.copy2(study/"selection_cpu_checks.log", destination/"validation_selection_cpu_checks.log")
    print({"selection_cpu_checks":cpu["tests"], "real_validation_ensembles":len(proof["checks"]),
           "max_metric_difference":max(row["max_metric_difference"] for row in proof["checks"]),
           "original_backtests":len(original_completed_backtests), "extra_restart_backtests":0,
           "official_selection_locked":False, "test_evaluated":False})


if __name__ == "__main__":
    summarize()
