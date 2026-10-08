"""Controlled per-day score normalization before baseline avg_none seed ensembles."""
from __future__ import annotations

import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research import ARTIFACTS, ROOT
from research.common import code_fingerprint, digest_file, now, write_json
from research.protocol import evaluate_predictions
from research.signals import blend_predictions

NAMES = ["csi300_ple16_no_raw_seeds012", "csi300_static_risk_seeds012",
         "sp500_raw_excess_seeds012", "sp500_independent_regime_seeds012",
         "csi300_mixture_1_seeds012", "csi300_mixture_2_seeds012"]
KEYS = ["IC", "ICIR", "RankIC", "RankICIR", "AR", "STD", "MDD", "Sharpe", "Sortino", "Calmar"]


def main():
    destination = ROOT / "research/records/20261008"
    original = json.loads((destination / "seed_validation_checks.json").read_text())["records"]
    records = []
    code = code_fingerprint()
    for name in NAMES:
        source = next(row for row in original if row["name"] == name)
        assert source["seeds"] == [0, 1, 2]
        frames, provenance = [], []
        for individual in source["individuals"]:
            folder = ARTIFACTS / "trials" / individual["id"]
            result = json.loads((folder / "result.json").read_text())
            assert result["status"] == "complete" and not result.get("smoke")
            assert result["config"] == individual["config"]
            frames.append(pd.read_pickle(folder / "valid/predictions.pkl"))
            provenance.append({"id": individual["id"], "config": individual["config"],
                               "prediction_sha256": digest_file(folder / "valid/predictions.pkl")})
        reference = frames[0]
        assert set(reference.index.get_level_values("datetime").year) == {2021, 2022}
        for frame in frames[1:]:
            pd.testing.assert_index_equal(frame.index, reference.index)
            np.testing.assert_equal(frame.label.to_numpy(), reference.label.to_numpy())
        for normalization in ["none", "cs_z"]:
            begin = time.monotonic()
            cpu_begin = resource.getrusage(resource.RUSAGE_SELF)
            transformed = [blend_predictions([frame], [1.], normalization, 1.0, 5) for frame in frames]
            for raw, frame in zip(frames, transformed):
                np.testing.assert_equal(raw.score.groupby(level="datetime").rank().to_numpy(),
                                        frame.score.groupby(level="datetime").rank().to_numpy())
                poisoned = raw.copy()
                poisoned["label"] = 123456789.
                np.testing.assert_equal(frame.score.to_numpy(),
                                        blend_predictions([poisoned], [1.], normalization, 1.0, 5).score.to_numpy())
            combined = reference.copy()
            combined["score"] = np.mean([frame.score.to_numpy() for frame in transformed], axis=0)
            case_name = f"{name}_{normalization}"
            output = ARTIFACTS / "validation_checks/score_normalization" / case_name
            identity = {"source": provenance, "normalization": normalization, "code": code,
                        "helper_sha256": digest_file(Path(__file__)), "method": "avg_none", "seeds": [0, 1, 2]}
            cached = json.loads((output / "ensemble.json").read_text()) if (output / "ensemble.json").exists() else {}
            if cached == identity and (output / "metrics.json").exists():
                metrics = json.loads((output / "metrics.json").read_text())
            else:
                metrics = evaluate_predictions(combined, source["market"], output, backtest=True)
                write_json(output / "ensemble.json", identity)
            cpu_end = resource.getrusage(resource.RUSAGE_SELF)
            record = {"name": case_name, "source_name": name, "market": source["market"],
                      "created_at": now(), "seeds": [0, 1, 2], "method": "avg_none",
                      "normalization_inside_candidate": normalization, "single_seed_ranks_exactly_unchanged": True,
                      "normalization_label_independent": True, "evaluation_code": code, "sources": provenance,
                      "validation": metrics, "seconds": time.monotonic() - begin,
                      "cpu_seconds": cpu_end.ru_utime + cpu_end.ru_stime - cpu_begin.ru_utime - cpu_begin.ru_stime,
                      "prediction_sha256": digest_file(output / "predictions.pkl"),
                      "original_float32_avg_none": source["ensemble"], "output": str(output)}
            records.append(record)
            comparisons = []
            for source_name in NAMES:
                pair = [r for r in records if r["source_name"] == source_name]
                if len(pair) == 2:
                    none, normalized = pair
                    comparisons.append({"source_name": source_name,
                                        "cs_z_minus_float64_identity": {key: normalized["validation"][key] - none["validation"][key] for key in KEYS},
                                        "float64_identity_minus_original": {key: none["validation"][key] - none["original_float32_avg_none"][key] for key in KEYS}})
            write_json(destination / "score_scale_validation_checks.json", {
                "created_at": now(), "scope": "purged validation 2021-2022 only; no new-model test",
                "planned_ensembles": 12, "completed_ensembles": len(records),
                "notes": ["All candidates retain baseline avg_none; normalization is inside each seed's forecast function.",
                          "Both identity and cs_z use the same existing float64 signal-combination code; the identity arm controls precision changes.",
                          "Daily normalization sees only current cross-sectional predictions; no labels or future dates.",
                          "These are controlled diagnostics, excluded from full trial counts and automated promotion."],
                "score_scale_probe": json.loads((ARTIFACTS / "study/score_scale_diagnostics.json").read_text()),
                "records": records, "comparisons": comparisons})
            print({"case": case_name, "validation": metrics}, flush=True)


if __name__ == "__main__":
    main()
