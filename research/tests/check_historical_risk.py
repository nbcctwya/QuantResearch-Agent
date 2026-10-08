"""Real validation prices and full zero-penalty backtests in both markets."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from research import ARTIFACTS, ROOT
from research.common import code_fingerprint, config_id, digest_file, now, write_json
from research.historical_risk import rolling_price_risk, verify_historical_source
from research.train import run_training


def main():
    plan = json.loads((ARTIFACTS/"study/historical_risk_design.json").read_text())
    assert code_fingerprint() == plan["code"]
    assert Path(__import__("research").__file__).parent == Path(plan["bundle"])/"research"
    keys = ["IC", "ICIR", "RankIC", "RankICIR", "AR", "STD", "MDD", "Sharpe", "Sortino", "Calmar"]
    rows = []
    for config in plan["zero_controls"]:
        identifier = config_id(config)
        source = ARTIFACTS/"trials"/config_id(config["alpha_source"])
        before = {name:(digest_file(source/name), (source/name).stat().st_mtime_ns)
                  for name in ["best.pt", "result.json", "valid/predictions.pkl"]}
        destination = ARTIFACTS/"historical_risk_proofs"/identifier
        frame = run_training(config, destination)
        original = pd.read_pickle(source/"valid/predictions.pkl")
        pd.testing.assert_frame_equal(frame, original, check_exact=True)
        output = pd.read_pickle(destination/"valid/predictions.pkl")
        pd.testing.assert_frame_equal(output, original, check_exact=True)
        assert before == {name:(digest_file(source/name), (source/name).stat().st_mtime_ns) for name in before}
        component = verify_historical_source(config, destination)
        cache = component["validation_price_cache"]
        prices = pd.read_pickle(Path(cache["path"])/"prices.pkl")
        benchmark = cache["benchmark"]
        names = sorted(set(prices.columns)-{benchmark})[:4]
        selected = prices[[benchmark]+names]
        risk = rolling_price_risk(selected, benchmark, 120, 60)
        # Independently solve the intercept OLS and residual risk on actual data.
        raw_prices = selected.to_numpy(dtype=np.float64)
        returns = raw_prices[1:]/raw_prices[:-1]-1
        verified = 0
        for day in [120, 180, len(selected)-1]:
            for column, name in enumerate(names, start=1):
                pairs = returns[day-120:day][:, [0, column]]
                pairs = pairs[np.isfinite(pairs).all(axis=1)]
                if len(pairs) < 60 or pairs[:, 0].var(ddof=1) <= 1e-12:
                    continue
                design = np.c_[np.ones(len(pairs)), pairs[:, 0]]
                fit = np.linalg.lstsq(design, pairs[:, 1], rcond=None)[0]
                residual = pairs[:, 1]-design@fit
                np.testing.assert_allclose(risk["beta"].iloc[day][name], fit[1], rtol=1e-10, atol=1e-11)
                np.testing.assert_allclose(risk["total_volatility"].iloc[day][name], pairs[:, 1].std(ddof=1), rtol=1e-10, atol=1e-11)
                np.testing.assert_allclose(risk["idiosyncratic_volatility"].iloc[day][name], np.sqrt(np.sum(residual**2)/(len(pairs)-1)), rtol=1e-10, atol=1e-11)
                verified += 1
        assert verified >= 8
        prefix_end = 240
        prefix = rolling_price_risk(selected.iloc[:prefix_end], benchmark, 120, 60)
        changed = selected.copy()
        changed.iloc[prefix_end:] *= np.arange(1, len(changed)-prefix_end+1)[:, None]**2
        poisoned = rolling_price_risk(changed, benchmark, 120, 60)
        for key in risk:
            pd.testing.assert_frame_equal(risk[key].iloc[:prefix_end], prefix[key], check_exact=True)
            pd.testing.assert_frame_equal(risk[key].iloc[:prefix_end], poisoned[key].iloc[:prefix_end], check_exact=True)
        metrics = json.loads((destination/"valid/metrics.json").read_text())
        old = json.loads((source/"valid/metrics.json").read_text())
        deltas = {key:metrics[key]-old[key] for key in keys}
        assert max(abs(value) for value in deltas.values()) < 1e-12
        first = pd.read_csv(destination/"valid/curve.csv")
        second = pd.read_csv(source/"valid/curve.csv")
        assert list(first.columns) == list(second.columns) and first.shape == second.shape
        assert first.iloc[:, 0].equals(second.iloc[:, 0])
        np.testing.assert_allclose(first.iloc[:, 1:], second.iloc[:, 1:], atol=1e-12, rtol=1e-12)
        rows.append({"market":config["market"], "id":identifier, "config":config,
                     "destination":str(destination), "prediction_exact":True, "alpha_artifacts_and_mtimes_unchanged":True,
                     "price_cache":cache, "independent_regressions":verified, "actual_price_prefix_and_future_poison_exact":True,
                     "metric_deltas":deltas, "baseline_curve_matches":True})
        write_json(ARTIFACTS/"study/historical_risk_route_checks.json", {"passed":len(rows)==2, "created_at":now(),
                   "code":code_fingerprint(), "cases_completed":len(rows), "scope":"both markets; complete purged validation; zero penalty, actual prices, no test",
                   "harness_sha256":digest_file(Path(__file__)), "experiments":rows})
        print({"market":config["market"], "actual_price_ols_cases":verified, "zero_predictions_exact":True, "metric_delta_max":max(abs(value) for value in deltas.values())}, flush=True)


if __name__ == "__main__":
    main()
