"""Actual training prices: independent sample risks, causal prefixes and targets."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research import ARTIFACTS
from research.common import code_fingerprint, digest_file, now, write_json
from research.data import DailyData
from research.historical_risk import rolling_price_risk
from research.market_targets import build_training_targets
from research.protocol import raw_labels
from research.volatility_targets import build_volatility_targets, training_volatility_data


def file_state(paths):
    return {str(path):{"sha256":digest_file(path),"mtime_ns":path.stat().st_mtime_ns} for path in paths}


def main():
    study = ARTIFACTS/"study"
    design = json.loads((study/"volatility_target_design.json").read_text())
    assert code_fingerprint() == design["code"]
    assert Path(__import__("research").__file__).parent == Path(design["bundle"])/"research"
    records = []
    for market in ["csi300","sp500"]:
        data = DailyData(market,"train",purge_days=5)
        index = data.index[data.selected_positions()]
        inputs = [ARTIFACTS/"cache"/market/"train"/name for name in ["features.npy","label.npy","index.pkl","manifest.json"]]
        input_before = file_state(inputs)
        raw = raw_labels(market,index).to_numpy(dtype=np.float32)
        configs = [config for config in design["candidates"] if config["market"] == market]
        features,metadata = training_volatility_data(data,configs[0])
        parent = metadata["parent_market_data"]
        folder = Path(parent["path"])
        source_paths = [folder/name for name in ["history_prices.pkl","benchmark_prices.pkl","features.pkl","manifest.json"]]
        parent_before = file_state(source_paths)
        history = pd.read_pickle(folder/"history_prices.pkl")
        market_data = pd.read_pickle(folder/"features.pkl")
        assert history.index.max() == index.get_level_values("datetime").max()
        assert pd.Timestamp(metadata["stock_price_end"]) <= pd.Timestamp("2020-12-31")
        returns = history.astype(np.float64).where(lambda value:np.isfinite(value)&(value>0)).pct_change(fill_method=None)
        checks = []
        usable = features.dropna()
        for position in np.linspace(0,len(usable)-1,12).astype(int):
            day,stock = usable.index[position]
            end = history.index.get_loc(day)
            sample = returns.iloc[max(0,end-119):end+1][[parent["benchmark"],stock]].dropna()
            regression = np.c_[np.ones(len(sample)),sample.iloc[:,0].to_numpy()]
            coefficients = np.linalg.lstsq(regression,sample.iloc[:,1].to_numpy(),rcond=None)[0]
            residuals = sample.iloc[:,1].to_numpy()-regression@coefficients
            expected_total = float(sample.iloc[:,1].std(ddof=1))
            # Match the documented sample-variance decomposition, RSS/(n-1).
            expected_idio = float(np.sqrt(residuals@residuals/(len(sample)-1)))
            observed = features.loc[(day,stock)]
            np.testing.assert_allclose(observed.total_volatility,expected_total,rtol=0,atol=1e-12)
            np.testing.assert_allclose(observed.idiosyncratic_volatility,expected_idio,rtol=0,atol=1e-12)
            checks.append({"date":str(day),"stock":stock,"observations":len(sample),
                "total_error":float(observed.total_volatility-expected_total),
                "idiosyncratic_error":float(observed.idiosyncratic_volatility-expected_idio)})
        original = rolling_price_risk(history,parent["benchmark"],120,60)
        cutoff = len(history)//2
        changed = history.copy()
        changed.iloc[cutoff:] *= np.linspace(2,40,len(history)-cutoff)[:,None]
        poisoned = rolling_price_risk(changed,parent["benchmark"],120,60)
        for name in ["total_volatility","idiosyncratic_volatility"]:
            pd.testing.assert_frame_equal(original[name].iloc[:cutoff],poisoned[name].iloc[:cutoff],check_exact=True)
        targets = []
        exposure = market_data.beta.to_numpy(dtype=np.float64)
        exposure = np.where(np.isfinite(exposure),exposure,1).clip(-3,3)
        for config in configs:
            actual,scale,proof = build_volatility_targets(data,config,raw)
            sigma = features[config["volatility_target_mode"]].to_numpy(dtype=np.float64)
            positive = np.sort(sigma[np.isfinite(sigma)&(sigma>0)])
            point = .05*(len(positive)-1)
            lo,hi = int(np.floor(point)),int(np.ceil(point))
            floor = max(float(positive[lo]+(positive[hi]-positive[lo])*(point-lo)),1e-6)
            median = float(np.median(positive))
            bounded = np.maximum(np.where(np.isfinite(sigma)&(sigma>=0),sigma,median),floor)
            base = raw.astype(np.float64)
            if config["volatility_target_base"] == "market_residual":
                base -= exposure*market_data.market_return.to_numpy()
            adjusted = base/np.power(bounded,config["volatility_target_power"])
            expected_scale = float(adjusted.std(ddof=1))
            expected = np.clip(adjusted.astype(np.float32)/expected_scale,-8,8).astype(np.float32)
            np.testing.assert_array_equal(actual,expected)
            assert scale == expected_scale
            assert proof["volatility_statistics"]["floor"] == floor
            assert proof["volatility_statistics"]["positive_median"] == median
            assert not proof["forecast_inputs_changed"] and not proof["test_targets_used"]
            targets.append({"config":config,"proof":proof,"independent_target_formula_exact":True})
        controls = []
        for base in ["raw","market_residual"]:
            config = next(config for config in configs if config["volatility_target_base"]==base)
            actual,scale,proof = build_volatility_targets(data,{**config,"volatility_target_power":0},raw)
            if base == "raw":
                from research.train import standardized_return_targets
                expected,old_scale = standardized_return_targets(raw,index,"raw_standardized")
            else:
                expected,old_scale,original_proof = build_training_targets(data,config,raw)
                assert proof["zero_power_base_signature"] == original_proof["target_signature"]
                stored = next(control for control in design["existing_controls"]
                    if control["config"]["market"]==market and control["config"]["target_kind"]=="market_residual_standardized")
                old_proof = json.loads((Path(stored["directory"])/"market_residual.json").read_text())
                assert original_proof["target_signature"] == old_proof["target_signature"]
            np.testing.assert_array_equal(actual,expected)
            assert scale == old_scale and proof["volatility_data"] is None
            controls.append({"base":base,"exact_old_targets_and_scale":True})
        cache = Path(metadata["path"])
        cache_before = file_state([cache/"features.pkl",cache/"manifest.json"])
        repeated,repeated_metadata = training_volatility_data(data,configs[0])
        pd.testing.assert_frame_equal(repeated,features,check_exact=True)
        assert metadata == repeated_metadata and cache_before == file_state([Path(name) for name in cache_before])
        assert input_before == file_state(inputs) and parent_before == file_state(source_paths)
        records.append({"market":market,"rows":len(index),"forecast_last":str(index[-1][0]),
            "independent_sample_and_ols_risk_checks":checks,"future_price_poison_prefix_exact":True,
            "target_cases":targets,"zero_power_controls":controls,"input_files_unchanged":input_before,
            "parent_price_files_unchanged":parent_before,"cache_restart_files_unchanged":cache_before})
        write_json(study/"volatility_target_data_checks.json",{"updated_at":now(),"passed":len(records)==2,
            "markets_completed":len(records),"scope":"complete purged training data only; no validation/test query",
            "records":records,"code":code_fingerprint(),"harness_sha256":digest_file(Path(__file__))})
        print({"market":market,"rows":len(index),"independent_risk_cases":12,"target_cases":4,
            "zero_controls":2,"source_files_unchanged":True},flush=True)


if __name__ == "__main__":
    main()
