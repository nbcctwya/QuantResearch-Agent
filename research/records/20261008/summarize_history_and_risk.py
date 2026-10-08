"""Export matched validation controls and their completed implementation audits."""
from __future__ import annotations

import json

import pandas as pd

from research import ARTIFACTS, ROOT
from research.common import digest_file, now, write_json

DESTINATION = ROOT / "research/records/20261008"
METRICS = ["IC", "ICIR", "RankIC", "RankICIR", "AR", "STD", "MDD", "Sharpe", "Sortino", "Calmar"]


def load(name):
    return json.loads((ARTIFACTS / "study" / name).read_text())


def completed(design, metadata_names):
    records = []
    for item in design["comparisons"]:
        folder = ARTIFACTS / "trials" / item["id"]
        if not (folder / "result.json").exists():
            continue
        result = json.loads((folder / "result.json").read_text())
        assert result["status"] == "complete" and not result.get("smoke")
        assert result["config"] == item["config"]
        record = {**item, "status": "complete", "validation": result["validation"],
                  "seconds": result["seconds"], "resources": result.get("resources"),
                  "run": json.loads((folder / "run.json").read_text()),
                  "checkpoint_sha256": digest_file(folder / "best.pt"),
                  "prediction_sha256": digest_file(folder / "valid/predictions.pkl")}
        for name in metadata_names:
            if (folder / name).exists():
                record[name] = json.loads((folder / name).read_text())
        records.append(record)
    return records


def summarize():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    history_design = load("history_design.json")
    histories = completed(history_design, ["history_model.json", "feature_encoder.json", "batching.json"])
    history_controls = {(r["market"], r["config"]["feature_encoder"]): r for r in histories
                        if r["history_steps"] == 8}
    for record in histories:
        control = history_controls.get((record["market"], record["config"]["feature_encoder"]))
        if control:
            record["matched_eight_steps"] = {"id": control["id"], "deltas": {
                key: record["validation"][key] - control["validation"][key] for key in METRICS}}
    write_json(DESTINATION / "history_validation_checks.json", {
        "created_at": now(), "scope": "purged validation 2021-2022; no new-model test outputs",
        "planned": len(history_design["comparisons"]), "completed": len(histories),
        "design": history_design, "cpu_checks": load("history_checks.json"),
        "native_cache_checks": load("history_cache_checks.json"),
        "legacy_prediction_checks": load("history_legacy_compatibility.json"),
        "gpu_smokes": load("history_gpu_checks.json"), "exact_gpu_recovery": load("history_resume_checks.json"),
        "notes": ["All smoke cases are excluded from full experiment counts and model selection.",
                  "Native histories preserve endpoint membership, latest features and baseline target.",
                  "Temporal mixing capacity changes with window length; this is an explicit experimental variable.",
                  "Original learn-row label eligibility and JKP release/revision timing remain unverified.",
                  "The legacy comparison covers two trained models, five validation days per market."],
        "experiments": histories})
    pd.DataFrame([{"id": r["id"], "market": r["market"], "history_steps": r["history_steps"],
                   "encoder": r["config"]["feature_encoder"], "seconds": r["seconds"], **r["validation"]}
                  for r in histories], columns=["id", "market", "history_steps", "encoder", "seconds", *METRICS]
                 ).to_csv(DESTINATION / "history_validation_metrics.csv", index=False)

    risk_design = load("mixture_risk_design.json")
    risks = completed(risk_design, ["fixed_alpha.json", "risk_source.json", "risk_regime.json",
                                    "valid/risk_regime/summary.json"])
    risk_controls = {(r["market"], r["risk_regime_strength"]): r for r in risks if r["risk_components"] == 1}
    for record in risks:
        control = risk_controls.get((record["market"], record["risk_regime_strength"]))
        if control:
            assert record["fixed_alpha.json"]["checkpoint_sha256"] == control["fixed_alpha.json"]["checkpoint_sha256"]
            record["matched_one_component"] = {"id": control["id"], "deltas": {
                key: record["validation"][key] - control["validation"][key] for key in METRICS}}
    findings = None
    if len(risks) == len(risk_design["comparisons"]):
        def values(market, components, strength):
            record = next(r for r in risks if r["market"] == market and r["risk_components"] == components
                          and r["risk_regime_strength"] == strength)
            return {key: record["validation"][key] for key in METRICS}
        findings = {
            "scope": "seed-0 validation; paired alpha seeds queued; risk source remains seed 0",
            "csi300_static": {"one_component": values("csi300", 1, 0.0),
                              "two_components": values("csi300", 2, 0.0)},
            "csi300_interpretation": "two components improve AR and ranking, with a small STD increase",
            "sp500_one_component": {"static": values("sp500", 1, 0.0),
                                    "regime": values("sp500", 1, 0.5)},
            "sp500_interpretation": "one-component regime scoring has the best RankIC, AR and STD within these controls; two-component regime scoring has slightly better MDD",
            "holdout_success_established": False}
    write_json(DESTINATION / "mixture_risk_validation_checks.json", {
        "created_at": now(), "scope": "purged validation 2021-2022; no new-model test outputs",
        "planned": len(risk_design["comparisons"]), "completed": len(risks), "design": risk_design,
        "gpu_recovery": load("mixture_risk_gpu_checks.json"),
        "full_zero_penalty_reproduction": load("mixture_risk_zero_reproduction.json"),
        "notes": ["The alpha checkpoint is identical across all risk-source controls within each market.",
                  "Risk sources are complete seed-0 models; only component count and static/regime scoring vary.",
                  "Density improvement is not evidence of improved portfolio performance.",
                  "Smoke/reproduction audits are excluded from search and promotion."],
        "current_findings": findings, "experiments": risks})
    pd.DataFrame([{"id": r["id"], "market": r["market"], "risk_components": r["risk_components"],
                   "regime_strength": r["risk_regime_strength"], "seconds": r["seconds"], **r["validation"]}
                  for r in risks], columns=["id", "market", "risk_components", "regime_strength", "seconds", *METRICS]
                 ).to_csv(DESTINATION / "mixture_risk_validation_metrics.csv", index=False)
    print({"history_controls_complete": len(histories), "mixture_risk_controls_complete": len(risks)})


if __name__ == "__main__":
    summarize()
