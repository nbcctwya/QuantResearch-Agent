"""Backtest actual paired three-seed mixture means on validation only."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from research import ARTIFACTS, ROOT
from research.common import code_fingerprint, config_id, digest_file, now, write_json
from research.protocol import evaluate_predictions

DESTINATION = ROOT / "research/records/20261008"
METRICS = ["IC", "ICIR", "RankIC", "RankICIR", "AR", "STD", "MDD", "Sharpe", "Sortino", "Calmar"]


def main():
    design = json.loads((ARTIFACTS / "study/mixture_design.json").read_text())
    records, pending = [], []
    for variant in ["one_component", "two_components"]:
        control = next(item for item in design["comparisons"] if item["market"] == "csi300"
                       and item["variant"] == variant)
        configurations = [{**control["config"], "seed": seed} for seed in [0, 1, 2]]
        missing = [config_id(c) for c in configurations
                   if not (ARTIFACTS / "trials" / config_id(c) / "result.json").exists()]
        if missing:
            pending.append({"variant": variant, "missing": missing})
            continue
        frames, individuals = [], []
        for configuration in configurations:
            identifier = config_id(configuration)
            folder = ARTIFACTS / "trials" / identifier
            result = json.loads((folder / "result.json").read_text())
            assert result["status"] == "complete" and not result.get("smoke")
            assert result["config"] == configuration
            frames.append(pd.read_pickle(folder / "valid/predictions.pkl"))
            individuals.append({"id": identifier, "config": configuration, "metrics": result["validation"],
                                "prediction_sha256": digest_file(folder / "valid/predictions.pkl"),
                                "checkpoint_sha256": digest_file(folder / "best.pt"),
                                "run": json.loads((folder / "run.json").read_text())})
        reference = frames[0]
        assert set(reference.index.get_level_values("datetime").year) == {2021, 2022}
        for frame in frames[1:]:
            pd.testing.assert_index_equal(frame.index, reference.index)
            np.testing.assert_equal(frame.label.to_numpy(), reference.label.to_numpy())
        combined = reference.copy()
        combined["score"] = np.mean([frame.score.to_numpy() for frame in frames], axis=0)
        name = f"csi300_mixture_{control['config']['mixture_components']}_seeds012"
        output = ARTIFACTS / "validation_checks" / name
        provenance = {"method": "avg_none", "seeds": [0, 1, 2], "code": code_fingerprint(),
                      "predictions": {r["id"]: r["prediction_sha256"] for r in individuals}}
        cached = json.loads((output / "ensemble.json").read_text()) if (output / "ensemble.json").exists() else {}
        if cached == provenance and (output / "metrics.json").exists():
            metrics = json.loads((output / "metrics.json").read_text())
        else:
            metrics = evaluate_predictions(combined, "csi300", output, backtest=True)
            write_json(output / "ensemble.json", provenance)
        records.append({"name": name, "created_at": now(), "market": "csi300", "variant": variant,
                        "split": "purged validation 2021-2022 only", "method": "avg_none", "seeds": [0, 1, 2],
                        "scope": "paired stability diagnostic; not an independent search candidate or holdout result",
                        "rows": len(combined), "ensemble": metrics, "individuals": individuals,
                        "ensemble_prediction_sha256": digest_file(output / "predictions.pkl"),
                        "code": provenance["code"], "output": str(output)})
        print({"completed": name, "validation": metrics}, flush=True)
    comparison = None
    if len(records) == 2:
        one, two = records
        deltas = {key: two["ensemble"][key] - one["ensemble"][key] for key in METRICS}
        comparison = {"deltas_two_minus_one": deltas,
                      "two_components_better": {key: value < 0 if key == "STD" else value > 0
                                                for key, value in deltas.items()},
                      "paired_seed_deltas": [{"seed": index, "deltas_two_minus_one": {
                          key: two["individuals"][index]["metrics"][key] - one["individuals"][index]["metrics"][key]
                          for key in METRICS}} for index in range(3)]}
    write_json(DESTINATION / "mixture_seed_validation_checks.json", {
        "created_at": now(), "scope": "CSI300 paired components; validation only; test remains unopened",
        "completed_ensembles": len(records), "records": records, "pending": pending, "comparison": comparison})
    if records:
        existing_path = DESTINATION / "seed_validation_checks.json"
        existing = json.loads(existing_path.read_text())
        prior = {row["name"] for row in existing["records"]}
        existing["records"].extend(row for row in records if row["name"] not in prior)
        existing["updated_at"] = now()
        write_json(existing_path, existing)


if __name__ == "__main__":
    main()
