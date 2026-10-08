"""Verify append-only experiment evidence and actual immutable launch packages."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess

from research import ARTIFACTS, ROOT
from research.common import digest_file, now, write_json

DESTINATION = ROOT / "research/records/20261008"


def verify(previous_commit):
    def previous(name):
        return subprocess.check_output(["git", "show", f"{previous_commit}:research/records/20261008/{name}"],
                                       cwd=ROOT, text=True)
    old = json.loads(previous("validation_runs.json"))["experiments"]
    new = json.loads((DESTINATION / "validation_runs.json").read_text())["experiments"]
    assert all(new[identifier] == record for identifier,record in old.items()), "Prior run metadata changed"
    def rows(value):
        return {row["id"]:row for row in csv.DictReader(io.StringIO(value))}
    before_rows = rows(previous("validation_metrics.csv"))
    after_rows = rows((DESTINATION / "validation_metrics.csv").read_text())
    assert all(after_rows[identifier] == row for identifier,row in before_rows.items()), "Prior metric row changed"
    before_seeds = json.loads(previous("seed_validation_checks.json"))["records"]
    after_seeds = json.loads((DESTINATION / "seed_validation_checks.json").read_text())["records"]
    assert after_seeds[:len(before_seeds)] == before_seeds, "Prior seed ensemble record changed"
    assert len({row["name"] for row in after_seeds}) == len(after_seeds), "Duplicate ensemble names"
    manifests, packages = {}, []
    for folder in sorted((DESTINATION / "code_snapshots").iterdir()):
        manifest = json.loads((folder / "manifest.json").read_text())
        files = manifest["files"]
        assert all(digest_file(folder / name) == expected for name,expected in files.items()), folder.name
        identifier = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:24]
        assert identifier == folder.name
        manifests[folder.name] = files
        packages.append({"bundle":folder.name, "manifest_sha256":digest_file(folder / "manifest.json"), "files":len(files)})
    covered = 0
    for identifier,record in new.items():
        source = Path(record["run"].get("source_package", ""))
        if source.parent.parent.name == "code_releases":
            assert manifests[source.parent.name] == record["run"]["code"], identifier
            covered += 1
    report = {"created_at":now(), "previous_commit":previous_commit, "previous_runs":len(old),
              "snapshot_runs":len(new), "previous_records_unchanged":True,
              "previous_metric_rows_exactly_unchanged":True, "previous_seed_ensemble_records_unchanged":True,
              "prior_seed_ensembles":len(before_seeds), "current_seed_ensembles":len(after_seeds),
              "checked_code_packages":packages, "every_source_matches_saved_manifest":True,
              "runs_with_exact_saved_package":covered, "earlier_runs_without_saved_package":len(new)-covered,
              "earlier_run_note":"Early runs without a recorded immutable launch package retain original source hashes; no package is attributed retroactively.",
              "selection_lock_exists":(ARTIFACTS / "study/selection_lock.json").exists(),
              "new_model_holdout_records":len(list((ARTIFACTS / "holdout").glob("*/test_run.json"))),
              "scope":"completed full validation only; diagnostics excluded; prior evidence preserved"}
    prior_audit = json.loads(previous("mixture_snapshot_integrity.json"))
    if "previous_coverage_audit" in prior_audit:
        report["previous_coverage_audit"] = prior_audit["previous_coverage_audit"]
    write_json(DESTINATION / "mixture_snapshot_integrity.json", report)
    print({"runs":len(new), "prior_runs_preserved":len(old), "packages_checked":len(packages),
           "explicitly_frozen_runs":covered, "earlier_without_package":len(new)-covered,
           "seed_ensembles":len(after_seeds), "holdout_records":report["new_model_holdout_records"]})
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous", default="HEAD")
    arguments = parser.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", arguments.previous], cwd=ROOT, text=True).strip()
    verify(commit)
