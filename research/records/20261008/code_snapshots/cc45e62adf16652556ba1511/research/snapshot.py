"""Export a small Git-trackable validation snapshot without datasets or checkpoints."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from . import ARTIFACTS, ROOT
from .common import now,write_json
from .runner import read_results


def snapshot(destination):
    timestamp = now()
    rows = []
    details = {}
    for result in sorted(read_results(),key=lambda r:r["id"]):
        config = result["config"]
        identifier = result["id"]
        folder = ARTIFACTS/"trials"/identifier
        run = json.loads((folder/"run.json").read_text())
        resources = result.get("resources") or {}
        rows.append({"id":identifier,"market":config["market"],"family":config["family"],"seed":config["seed"],
                     "split":"validation 2021-2022","purge_days":config.get("purge_days",5),
                     "selection_score":result["selection_score"],"seconds":result["seconds"],
                     "cpu_seconds":resources.get("cpu_seconds"),"peak_process_rss_mib":resources.get("peak_process_rss_mib"),
                     "gpu_peak_allocated_mib":resources.get("gpu_peak_allocated_mib"),
                     "config_json":json.dumps(config,sort_keys=True),**result["validation"]})
        details[identifier] = {"config":config,"completed_at":result["completed_at"],"run":run,
                               "resources":resources,"calibration":result.get("calibration")}
        package = Path(run.get("source_package",ROOT/"research"))
        if package.parent.parent == ARTIFACTS/"code_releases":
            saved = destination/"code_snapshots"/package.parent.name
            if not saved.exists():
                saved.parent.mkdir(parents=True,exist_ok=True)
                shutil.copytree(package.parent,saved,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
        for name in ["target_transform.json","market_residual.json","volatility_target.json","risk_source.json","components.json","historical_risk.json","batching.json","feature_encoder.json",
                     "covariance_model.json","mixture_model.json","return_distribution.json","history_model.json","weight_averaging.json","ranking_objective.json","risk_regime.json","fixed_alpha.json",
                     "valid/risk_regime/summary.json","alpha_opportunity.json",
                     "valid/alpha_opportunity/summary.json","valid/calibration/calibration.json"]:
            if (folder/name).exists():
                details[identifier][name] = json.loads((folder/name).read_text())
    destination.mkdir(parents=True,exist_ok=True)
    table = pd.DataFrame(rows)
    if len(table):
        table = table.sort_values(["market","selection_score"],ascending=[True,False])
    table.to_csv(destination/"validation_metrics.csv",index=False)
    write_json(destination/"validation_runs.json",{"created_at":timestamp,"experiments":details,
               "scope":"completed full validation runs only; all smoke runs excluded; no new-model test metrics",
               "resource_note":"older runs record duration/device only; new peak-memory counters are process-lifetime peaks"})
    return {"created_at":timestamp,"runs":len(rows),"directory":str(destination)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=ROOT/"research/records/20261008")
    args = parser.parse_args()
    print(json.dumps(snapshot(args.out),ensure_ascii=False))


if __name__ == "__main__":
    main()
