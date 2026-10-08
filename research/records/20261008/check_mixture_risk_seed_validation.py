"""Paired frozen-risk validation using actual alpha-seed prediction averages."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research import ARTIFACTS, ROOT
from research.common import code_fingerprint, config_id, digest_file, now, write_json
from research.protocol import evaluate_predictions
from research.risk_regime import frozen_alpha_config

DESTINATION = ROOT / "research/records/20261008"
METRICS = ["IC", "ICIR", "RankIC", "RankICIR", "AR", "STD", "MDD", "Sharpe", "Sortino", "Calmar"]
GROUPS = [("csi300", 1, 0.0), ("csi300", 2, 0.0),
          ("sp500", 1, 0.0), ("sp500", 1, 0.5)]


def main():
    design = json.loads((ARTIFACTS / "study/mixture_risk_design.json").read_text())
    records, pending = [], []
    code = code_fingerprint()
    for market, components, strength in GROUPS:
        control = next(item for item in design["comparisons"] if item["market"] == market
                       and item["risk_components"] == components
                       and item["risk_regime_strength"] == strength)
        configurations = [{**control["config"], "seed": seed} for seed in [0, 1, 2]]
        missing = [config_id(config) for config in configurations
                   if not (ARTIFACTS / "trials" / config_id(config) / "result.json").exists()]
        name = f"{market}_frozen_mixture_{components}_regime_{strength:g}_alpha_seeds012"
        if missing:
            pending.append({"name": name, "market": market, "missing": missing})
            continue
        frames, individuals = [], []
        for config in configurations:
            folder = ARTIFACTS / "trials" / config_id(config)
            result = json.loads((folder / "result.json").read_text())
            assert result["status"] == "complete" and not result.get("smoke")
            assert result["config"] == config
            alpha = json.loads((folder / "fixed_alpha.json").read_text())
            risk = json.loads((folder / "risk_source.json").read_text())
            effective_alpha = frozen_alpha_config(config)
            assert alpha["config"] == effective_alpha and alpha["config"]["seed"] == config["seed"]
            assert risk["config"] == config["risk_source"] and risk["config"]["seed"] == 0
            assert alpha["frozen"] and risk["frozen"]
            for source in [alpha, risk]:
                checkpoint = ARTIFACTS / "trials" / config_id(source["config"]) / "best.pt"
                assert digest_file(checkpoint) == source["checkpoint_sha256"]
            frames.append(pd.read_pickle(folder / "valid/predictions.pkl"))
            individuals.append({"id": config_id(config), "config": config, "metrics": result["validation"],
                                "prediction_sha256": digest_file(folder / "valid/predictions.pkl"),
                                "checkpoint_sha256": digest_file(folder / "best.pt"),
                                "run": json.loads((folder / "run.json").read_text()),
                                "fixed_alpha": alpha, "frozen_risk": risk})
        reference = frames[0]
        assert set(reference.index.get_level_values("datetime").year) == {2021, 2022}
        for frame in frames[1:]:
            pd.testing.assert_index_equal(frame.index, reference.index)
            np.testing.assert_equal(frame.label.to_numpy(), reference.label.to_numpy())
        combined = reference.copy()
        combined["score"] = np.mean([frame.score.to_numpy() for frame in frames], axis=0)
        output = ARTIFACTS / "validation_checks" / name
        provenance = {"method": "avg_none", "alpha_seeds": [0, 1, 2], "risk_seed": 0,
                      "code": code, "helper_sha256": digest_file(Path(__file__)),
                      "sources": [{"id": row["id"], "prediction": row["prediction_sha256"],
                                   "checkpoint": row["checkpoint_sha256"],
                                   "alpha_checkpoint": row["fixed_alpha"]["checkpoint_sha256"],
                                   "risk_checkpoint": row["frozen_risk"]["checkpoint_sha256"]}
                                  for row in individuals]}
        cache_path = output / "ensemble.json"
        cached = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        if cached.get("provenance") == provenance and (output / "metrics.json").exists():
            assert cached["prediction_sha256"] == digest_file(output / "predictions.pkl")
            assert cached["metrics_sha256"] == digest_file(output / "metrics.json")
            metrics = json.loads((output / "metrics.json").read_text())
        else:
            metrics = evaluate_predictions(combined, market, output, backtest=True)
            write_json(cache_path, {"provenance": provenance,
                                   "prediction_sha256": digest_file(output / "predictions.pkl"),
                                   "metrics_sha256": digest_file(output / "metrics.json")})
        records.append({"name": name, "created_at": now(), "market": market,
                        "risk_components": components, "risk_regime_strength": strength,
                        "split": "purged validation 2021-2022 only", "method": "avg_none",
                        "seeds": [0, 1, 2], "alpha_seeds": [0, 1, 2], "risk_seed": 0,
                        "scope": "paired stability diagnostic; frozen risk seed is shared; no holdout result",
                        "rows": len(combined), "ensemble": metrics, "individuals": individuals,
                        "ensemble_prediction_sha256": digest_file(output / "predictions.pkl"),
                        "code": code, "output": str(output)})
        print({"completed": name, "validation": metrics}, flush=True)
    comparisons = []
    for market in ["csi300", "sp500"]:
        pair = [record for record in records if record["market"] == market]
        if len(pair) != 2:
            continue
        first, second = pair
        for one, two in zip(first["individuals"], second["individuals"]):
            assert one["fixed_alpha"]["checkpoint_sha256"] == two["fixed_alpha"]["checkpoint_sha256"]
            if market == "sp500":
                assert one["frozen_risk"]["checkpoint_sha256"] == two["frozen_risk"]["checkpoint_sha256"]
        deltas = {key: second["ensemble"][key] - first["ensemble"][key] for key in METRICS}
        comparisons.append({"market": market, "control": first["name"], "comparison": second["name"],
                            "deltas_comparison_minus_control": deltas,
                            "comparison_better": {key: value < 0 if key == "STD" else value > 0
                                                  for key, value in deltas.items()},
                            "paired_seed_deltas": [{"alpha_seed": seed, "risk_seed": 0,
                                "deltas_comparison_minus_control": {
                                    key: second["individuals"][seed]["metrics"][key]
                                         - first["individuals"][seed]["metrics"][key] for key in METRICS}}
                                for seed in range(3)]})
    write_json(DESTINATION / "mixture_risk_seed_validation_checks.json", {
        "created_at": now(), "scope": "paired frozen-risk scoring; validation only; test remains unopened",
        "planned_ensembles": 4, "completed_ensembles": len(records),
        "notes": ["Risk sources retain seed 0; only alpha models use paired seeds 0/1/2.",
                  "Actual avg_none predictions are backtested; individual seed metrics are never averaged.",
                  "These stability diagnostics are excluded from full trial counts and automated promotion."],
        "records": records, "pending": pending, "comparisons": comparisons})
    if records:
        existing_path = DESTINATION / "seed_validation_checks.json"
        existing = json.loads(existing_path.read_text())
        prior = {row["name"] for row in existing["records"]}
        additional = [row for row in records if row["name"] not in prior]
        if additional:
            existing["records"].extend(additional)
            existing["updated_at"] = now()
            write_json(existing_path, existing)
    print({"completed_ensembles": len(records), "pending_groups": len(pending)})


if __name__ == "__main__":
    main()
