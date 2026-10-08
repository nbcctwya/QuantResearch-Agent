"""Completed validation performance plus independently checked EMA implementation state."""
from __future__ import annotations

import json
import importlib

import pandas as pd
import torch

from research import ARTIFACTS, ROOT
from research.common import config_id, digest_file, now, write_json


def summarize():
    destination = ROOT / "research/records/20261008"
    study = ARTIFACTS / "study"
    design = json.loads((study / "ema_design.json").read_text())
    completed = importlib.import_module("research.records.20261008.summarize_history_and_risk").completed
    records = completed(design, ["weight_averaging.json", "feature_encoder.json", "target_transform.json", "batching.json"])
    controls = {r["market"]: r for r in records if r["decay"] is None}
    keys = ["IC", "ICIR", "RankIC", "RankICIR", "AR", "STD", "MDD", "Sharpe", "Sortino", "Calmar"]
    for record in records:
        control = controls.get(record["market"])
        if control:
            record["matched_unaveraged"] = {"id": control["id"], "deltas": {
                key: record["validation"][key] - control["validation"][key] for key in keys}}
    checks = {}
    for name in ["ema_cpu_checks", "ema_gpu_checks", "ema_resume_checks", "ema_legacy_gpu_checks"]:
        checks[name] = json.loads((study / f"{name}.json").read_text())
        assert checks[name]["passed"], name
    full_legacy = []
    for control in controls.values():
        original_config = {key:value for key,value in control["config"].items() if key != "ema_decay"}
        original = ARTIFACTS / "trials" / config_id(original_config)
        current = ARTIFACTS / "trials" / control["id"]
        before_result = json.loads((original / "result.json").read_text())
        assert before_result["status"] == "complete" and before_result["config"] == original_config
        pd.testing.assert_frame_equal(pd.read_pickle(original / "valid/predictions.pkl"),
                                      pd.read_pickle(current / "valid/predictions.pkl"), check_exact=True)
        before = torch.load(original / "best.pt", map_location="cpu", weights_only=False)
        after = torch.load(current / "best.pt", map_location="cpu", weights_only=False)
        assert before["epoch"] == after["epoch"] and set(before["model"]) == set(after["model"])
        for key,value in before["model"].items():
            torch.testing.assert_close(value, after["model"][key], rtol=0, atol=0)
        old_epochs = [json.loads(line) for line in (original / "epochs.jsonl").read_text().splitlines()]
        new_epochs = [json.loads(line) for line in (current / "epochs.jsonl").read_text().splitlines()]
        assert len(old_epochs) == len(new_epochs)
        for first,second in zip(old_epochs,new_epochs):
            for key in ["epoch", "train_loss", "valid_selection_score", "best_selection_score", "valid", "optimizer_steps"]:
                assert first[key] == second[key], (control["market"], key)
        full_legacy.append({"market":control["market"], "original_id":config_id(original_config),
                            "explicit_disabled_id":control["id"], "exact_predictions":True,
                            "exact_selected_parameters_and_epoch":True, "exact_training_losses":True,
                            "original_prediction_sha256":digest_file(original / "valid/predictions.pkl"),
                            "current_prediction_sha256":digest_file(current / "valid/predictions.pkl"),
                            "max_metric_difference":max(abs(control["validation"][key]-before_result["validation"][key]) for key in keys),
                            "scope":"complete training and purged validation; no holdout"})
    write_json(destination / "ema_validation_checks.json", {
        "created_at": now(), "scope": "purged validation only; GPU smoke and recovery excluded from full trial counts",
        "planned": len(design["comparisons"]), "completed": len(records), "design": design, "checks": checks,
        "full_disabled_ema_reproduction":full_legacy,
        "harnesses": {name: digest_file(ROOT / "research/tests" / name)
                      for name in ["check_ema_gpu.py", "check_ema_legacy.py", "check_mixture_recovery.py"]},
        "notes": ["Optimized raw parameters and averaged validation/inference parameters are distinct.",
                  "Train-fitted buffers remain exact; EMA introduces no validation/test fitting.",
                  "Disabled-EMA real GPU controls reproduce prior parameters, losses and predictions exactly.",
                  "Changing early-stop epoch is an explicit part of this training method.",
                  "Correct recovery is not evidence of improved portfolio results."], "experiments": records})
    pd.DataFrame([{"id": r["id"], "market": r["market"], "decay": r["decay"],
                   "seconds": r["seconds"], **r["validation"]} for r in records],
                 columns=["id", "market", "decay", "seconds", *keys]
                 ).to_csv(destination / "ema_validation_metrics.csv", index=False)
    scale = json.loads((destination / "score_scale_validation_checks.json").read_text())
    pd.DataFrame([{"name": r["name"], "source_name": r["source_name"], "market": r["market"],
                   "normalization": r["normalization_inside_candidate"], "seconds": r["seconds"],
                   "cpu_seconds": r["cpu_seconds"], **r["validation"]} for r in scale["records"]]
                 ).to_csv(destination / "score_scale_validation_metrics.csv", index=False)
    write_json(destination / "score_scale_method_design.json", json.loads((study / "score_scale_design.json").read_text()))
    print({"full_ema_controls_complete": len(records), "score_scale_ensembles": scale["completed_ensembles"]})


if __name__ == "__main__":
    summarize()
