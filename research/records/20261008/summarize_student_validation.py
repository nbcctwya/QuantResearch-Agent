"""Matched full Student-t controls; implementation checks retain their own scope."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import shutil

import pandas as pd

from research import ARTIFACTS, ROOT
from research.common import digest_file, now, write_json


def summarize():
    destination = ROOT / "research/records/20261008"
    study = ARTIFACTS / "study"
    design = json.loads((study / "student_design.json").read_text())
    completed = importlib.import_module("research.records.20261008.summarize_history_and_risk").completed
    records = completed(design, ["return_distribution.json", "mixture_model.json", "target_transform.json",
                                 "batching.json", "valid/calibration/calibration.json"])
    controls = {(row["market"], row["target_kind"]): row for row in records if row["existing_control"]}
    keys = ["IC", "ICIR", "RankIC", "RankICIR", "AR", "STD", "MDD", "Sharpe", "Sortino", "Calmar"]
    for row in records:
        control = controls.get((row["market"], row["target_kind"]))
        if control and not row["existing_control"]:
            assert row["target_transform.json"] == control["target_transform.json"]
            row["matched_gaussian_result"] = {"id":control["id"], "deltas":{
                key:row["validation"][key] - control["validation"][key] for key in keys}}
    checks = {}
    for name in ["student_cpu_checks", "student_gpu_checks", "student_resume_checks"]:
        checks[name] = json.loads((study / f"{name}.json").read_text())
        assert checks[name]["passed"] and checks[name]["code"] == design["code"], name
    assert checks["student_gpu_checks"]["completed"] == 2
    assert checks["student_resume_checks"]["cases_completed"] == 2
    # This is the package actually used by the passing GPU checks, even when full
    # Student runs are still pending. It is not counted as a full experiment.
    bundle = Path(design["bundle"])
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["files"] == design["code"]
    assert all(digest_file(bundle / name) == expected for name, expected in manifest["files"].items())
    saved = destination / "code_snapshots" / bundle.name
    if not saved.exists():
        shutil.copytree(bundle, saved, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    write_json(destination / "student_validation_checks.json", {
        "created_at":now(), "scope":"purged full validation only; implementation smokes excluded",
        "planned_comparisons":16, "completed_comparisons":len(records),
        "new_student_runs_completed":sum(not row["existing_control"] for row in records),
        "reused_gaussian_controls":sum(row["existing_control"] for row in records),
        "design":design, "checks":checks,
        "initial_cpu_check":json.loads((study / "student_cpu_checks_initial.json").read_text()),
        "harnesses":{name:digest_file(ROOT / "research/tests" / name)
                      for name in ["check_student_gpu.py", "test_student.py", "check_mixture_recovery.py"]},
        "notes":["One-component Gaussian controls retain their actual previous results and float64 likelihood.",
                 "The output log variance denotes total variance; Student-t scale is smaller by sqrt((df-2)/df).",
                 "Initial diagnostic precision failure is preserved; raw-unit Student moments and quantiles now use float64.",
                 "Correct likelihoods, intervals and recovery do not prove improved portfolio metrics.",
                 "Original data-eligibility and JKP revision/release-time caveats remain applicable."],
        "experiments":records})
    pd.DataFrame([{"id":row["id"], "market":row["market"], "target_kind":row["target_kind"],
                   "distribution":row["distribution"], "student_df":row["student_df"],
                   "reused_gaussian_control":row["existing_control"], "seconds":row["seconds"],
                   **row["validation"]} for row in records]).to_csv(destination / "student_validation_metrics.csv", index=False)
    print({"new_student_runs_completed":sum(not row["existing_control"] for row in records),
           "existing_matched_gaussian_controls":sum(row["existing_control"] for row in records)})


if __name__ == "__main__":
    summarize()
