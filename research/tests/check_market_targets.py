"""Real training-data checks of label horizons, past OLS, and unchanged inputs."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from qlib.data import D

from research import ARTIFACTS
from research.common import code_fingerprint, digest_file, now, write_json
from research.data import DailyData
from research.historical_risk import rolling_price_risk
from research.market_targets import build_training_targets, training_market_data
from research.protocol import initialize_qlib, raw_labels
from research.train import standardized_return_targets


def file_state(path):
    return {"sha256":digest_file(path), "mtime_ns":path.stat().st_mtime_ns}


def main():
    stage = json.loads((ARTIFACTS/"study/market_residual_stage.json").read_text())
    assert code_fingerprint() == stage["code"]
    assert Path(__import__("research").__file__).parent == Path(stage["bundle"])/"research"
    records = []
    for market in ["csi300", "sp500"]:
        data = DailyData(market, "train", purge_days=5)
        positions = data.selected_positions()
        index = data.index[positions]
        unchanged = {ARTIFACTS/"cache"/market/"train"/name:None
                     for name in ["features.npy", "label.npy", "index.pkl", "manifest.json"]}
        unchanged = {str(path):file_state(path) for path in unchanged}
        raw = raw_labels(market, index).to_numpy(dtype=np.float32)
        config = {"market":market, "family":"residual", "target_kind":"market_residual_standardized",
                  "market_residual_strength":1.0}
        features, metadata = training_market_data(data, config)
        folder = Path(metadata["path"])
        history = pd.read_pickle(folder/"history_prices.pkl")
        benchmark = pd.read_pickle(folder/"benchmark_prices.pkl")
        assert history.index.max() == index.get_level_values("datetime").max()
        assert benchmark.index.max() <= pd.Timestamp("2020-12-31")
        assert history.index.max() < benchmark.index.max()
        assert features.index.equals(index)
        initialize_qlib(market)
        expression = D.features([metadata["benchmark"]],
            ["Ref($close, -5) / Ref($close, -1) - 1"], start_time=index[0][0],
            end_time=index[-1][0], freq="day").iloc[:,0].droplevel("instrument")
        actual = features.market_return.groupby(level="datetime").first()
        expression = expression.reindex(actual.index)
        # Qlib computes from float32 close; our explicit endpoint ratio uses float64.
        np.testing.assert_allclose(actual, expression, rtol=0, atol=2e-7)
        horizon_error = float(np.abs(actual.to_numpy()-expression.to_numpy()).max())
        estimates = rolling_price_risk(history, metadata["benchmark"], 120, 60)
        returns = history.astype(np.float64).where(lambda value:np.isfinite(value)&(value>0)).pct_change(fill_method=None)
        pairs = features.beta.dropna()
        checks = []
        for position in np.linspace(0, len(pairs)-1, 12).astype(int):
            day, stock = pairs.index[position]
            end = history.index.get_loc(day)
            sample = returns.iloc[max(0,end-119):end+1][[metadata["benchmark"], stock]].dropna()
            design = np.c_[np.ones(len(sample)), sample.iloc[:,0].to_numpy()]
            expected = np.linalg.lstsq(design, sample.iloc[:,1].to_numpy(), rcond=None)[0][1]
            observed = features.loc[(day,stock), "beta"]
            np.testing.assert_allclose(observed, expected, rtol=0, atol=3e-12)
            checks.append({"date":str(day),"stock":stock,"observations":len(sample),
                           "slope_error":float(observed-expected)})
        cutoff = len(history)//2
        poisoned = history.copy()
        poisoned.iloc[cutoff:] *= np.linspace(2.,50.,len(history)-cutoff)[:,None]
        changed = rolling_price_risk(poisoned, metadata["benchmark"], 120, 60)
        pd.testing.assert_frame_equal(estimates["beta"].iloc[:cutoff],
            changed["beta"].iloc[:cutoff], check_exact=True)
        exposure = features.beta.to_numpy(dtype=np.float64)
        exposure = np.where(np.isfinite(exposure),exposure,1.).clip(-3.,3.)
        results = []
        for strength in [0., .5, 1.]:
            targets, scale, proof = build_training_targets(data,
                {**config,"market_residual_strength":strength},raw)
            reference = raw.astype(np.float64)-strength*exposure*features.market_return.to_numpy()
            expected_scale = float(reference.std(ddof=1))
            expected = np.clip(reference.astype(np.float32)/expected_scale,-8,8).astype(np.float32)
            np.testing.assert_array_equal(targets, expected)
            assert scale == expected_scale
            if strength == 0:
                legacy, old_scale = standardized_return_targets(raw,index,"raw_standardized")
                np.testing.assert_array_equal(targets,legacy)
                assert scale == old_scale and proof["market_data"] is None
            results.append({"strength":strength,"scale":scale,"proof":proof,
                            "independent_target_formula_exact":True})
        before = {str(path):file_state(path) for path in folder.iterdir() if path.is_file()}
        repeated, repeated_metadata = training_market_data(data,config)
        pd.testing.assert_frame_equal(features,repeated,check_exact=True)
        assert metadata == repeated_metadata
        assert before == {name:file_state(Path(name)) for name in before}
        assert unchanged == {name:file_state(Path(name)) for name in unchanged}
        record = {"market":market,"rows":len(index),"forecast_first":str(index[0][0]),
            "forecast_last":str(index[-1][0]),"benchmark_forward_last":str(benchmark.index.max()),
            "past_beta_ols_checks":checks,"future_poison_beta_prefix_exact":True,
            "qlib_horizon_max_error":horizon_error,"qlib_float32_tolerance":2e-7,
            "targets":results,"source_inputs_unchanged":unchanged,
            "cache_restart_hashes_and_mtimes_unchanged":before}
        records.append(record)
        write_json(ARTIFACTS/"study/market_residual_data_checks.json",{
            "updated_at":now(),"passed":len(records)==2,"markets_completed":len(records),
            "code":code_fingerprint(),"harness_sha256":digest_file(Path(__file__)),
            "scope":"actual purged training data only; no validation/test data queried",
            "records":records})
        print({"market":market,"rows":len(index),"independent_ols_cases":len(checks),
               "forward_end":str(benchmark.index.max()),"targets_exact":True},flush=True)


if __name__ == "__main__":
    main()
