"""Independent price-regression, causal-prefix, scoring and holdout-lock checks."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import research.historical_risk as historical
import research.train as training
from research.common import config_id, digest_file, write_json
from research.historical_risk import (aligned_risk, apply_historical_risk, load_price_risk,
                                      risk_spec, risk_z, rolling_price_risk, verify_historical_source)
from research.risk_regime import frozen_alpha_config


class HistoricalRiskTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(47)
        returns = rng.normal(0, .015, (180, 4))
        returns[:, 1] = 1.7*returns[:, 0]+rng.normal(0, .01, 180)
        returns[:, 2] = -.4*returns[:, 0]+rng.normal(0, .005, 180)
        self.prices = pd.DataFrame(100*np.cumprod(1+returns, axis=0),
            index=pd.bdate_range("2020-01-01", periods=180), columns=["MARKET", "A", "B", "C"])

    def test_rolling_ols_matches_independent_lstsq_with_missing_and_sample_variances(self):
        prices = self.prices.copy()
        prices.iloc[[51, 52, 110], 1] = np.nan
        prices.iloc[[80, 81], 0] = np.nan
        actual = rolling_price_risk(prices, "MARKET", 60, 30)
        returns = prices.to_numpy()[1:]/prices.to_numpy()[:-1]-1
        for day in [50, 90, 130, 179]:
            for column, name in enumerate(prices.columns[1:], start=1):
                samples = returns[max(0, day-60):day][:, [0, column]]
                samples = samples[np.isfinite(samples).all(axis=1)]
                fit = np.linalg.lstsq(np.c_[np.ones(len(samples)), samples[:, 0]], samples[:, 1], rcond=None)[0]
                residual = samples[:, 1]-np.c_[np.ones(len(samples)), samples[:, 0]]@fit
                self.assertAlmostEqual(actual["beta"].iloc[day][name], fit[1], places=11)
                self.assertAlmostEqual(actual["total_volatility"].iloc[day][name], samples[:, 1].std(ddof=1), places=12)
                self.assertAlmostEqual(actual["idiosyncratic_volatility"].iloc[day][name], np.sqrt(np.sum(residual**2)/(len(samples)-1)), places=12)
                self.assertEqual(actual["observations"].iloc[day][name], len(samples))

    def test_future_price_poisoning_cannot_change_prefix_features_or_scores(self):
        first = rolling_price_risk(self.prices, "MARKET", 60, 30)
        changed = self.prices.copy()
        changed.iloc[121:] *= np.arange(1, len(changed)-120)[:, None]**2
        second = rolling_price_risk(changed, "MARKET", 60, 30)
        prefix = rolling_price_risk(self.prices.iloc[:121], "MARKET", 60, 30)
        for key in first:
            pd.testing.assert_frame_equal(first[key].iloc[:121], second[key].iloc[:121])
            pd.testing.assert_frame_equal(first[key].iloc[:121], prefix[key])

    def test_no_price_fill_and_insufficient_or_degenerate_market_is_neutral(self):
        prices = self.prices.copy()
        prices.iloc[100:140, 1] = np.nan
        result = rolling_price_risk(prices, "MARKET", 20, 15)
        self.assertEqual(result["observations"].iloc[135]["A"], 0)
        self.assertTrue(np.isnan(result["beta"].iloc[135]["A"]))
        prices["MARKET"] = 100
        result = rolling_price_risk(prices, "MARKET", 20, 15)
        self.assertTrue(result["beta"].isna().all().all())
        self.assertTrue(result["idiosyncratic_volatility"].isna().all().all())
        self.assertTrue(result["total_volatility"].iloc[50:].notna().any().any())

    def test_alignment_missing_stock_and_label_poisoning_preserve_scores(self):
        index = pd.MultiIndex.from_product([self.prices.index[100:103], ["A", "B", "C", "ABSENT"]], names=["datetime", "instrument"]).sort_values()
        risks = aligned_risk(rolling_price_risk(self.prices, "MARKET", 60, 30), index)
        self.assertTrue(risks.index.equals(index))
        self.assertTrue(risks.xs("ABSENT", level="instrument").isna().all().all())
        frame = pd.DataFrame({"score":np.arange(len(index), dtype=np.float32), "label":np.linspace(-1, 1, len(index))}, index=index)
        config = {"risk_penalty":.5}
        first = apply_historical_risk(frame, risks, config)
        frame["label"] = np.nan
        second = apply_historical_risk(frame, risks, config)
        np.testing.assert_array_equal(first.score, second.score)
        self.assertTrue(np.isfinite(first.score).all())
        self.assertEqual(risk_z(risks, risk_spec(config)).xs("ABSENT", level="instrument").sum(), 0)
        with self.assertRaises(ValueError):
            apply_historical_risk(frame, risks.iloc[::-1], config)

    def test_zero_penalty_is_exact_original_dtype_scores_and_labels(self):
        index = pd.MultiIndex.from_product([pd.to_datetime(["2021-01-04"]), ["A", "B"]], names=["datetime", "instrument"])
        frame = pd.DataFrame({"score":np.array([.12345678, .12345679], dtype=np.float32), "label":[.1, -.1]}, index=index)
        features = pd.DataFrame({"beta":[np.nan, np.nan]}, index=index)
        pd.testing.assert_frame_equal(apply_historical_risk(frame, features, {"risk_penalty":0}), frame, check_exact=True)

    def test_z_scores_are_per_date_and_clip_extreme_or_missing_risk(self):
        index = pd.MultiIndex.from_product([pd.to_datetime(["2021-01-04", "2021-01-05"]), list(map(str, range(30)))], names=["datetime", "instrument"])
        features = pd.DataFrame({"beta":np.r_[np.ones(29), 1000, np.repeat(np.nan, 30)]}, index=index)
        values = risk_z(features, risk_spec({}))
        self.assertEqual(values.iloc[29], 3)
        self.assertEqual(values.iloc[30:].sum(), 0)
        features["beta"] = 1
        self.assertEqual(risk_z(features, risk_spec({})).abs().sum(), 0)

    def test_invalid_specifications_and_unlocked_price_queries_fail(self):
        invalid = [{"risk_window":True}, {"risk_min_observations":1}, {"risk_window":10},
                   {"risk_penalty":True}, {"risk_penalty":float("nan")}, {"risk_penalty":-1},
                   {"historical_risk_mode":"future"}, {"alpha_norm":"none"}]
        for config in invalid:
            with self.assertRaises(ValueError):
                risk_spec(config)
        index = pd.MultiIndex.from_product([pd.to_datetime(["2023-01-04"]), ["A"]], names=["datetime", "instrument"])
        config = {"market":"sp500", "seed":0, "family":"historical_risk"}
        with self.assertRaises(ValueError):
            load_price_risk(config, index, "valid")
        with tempfile.TemporaryDirectory() as directory, patch.object(training, "ARTIFACTS", Path(directory)):
            with self.assertRaises(RuntimeError):
                load_price_risk(config, index, "test")

    def test_selected_historical_method_and_nested_blend_lock_only_its_seed_components(self):
        alpha = {"market":"sp500", "seed":0, "family":"residual", "purge_days":5}
        method = {"market":"sp500", "seed":0, "family":"historical_risk", "purge_days":5, "alpha_source":alpha, "risk_penalty":.5}
        blend = {"market":"sp500", "seed":0, "family":"scores_blend", "purge_days":5, "sources":[method, alpha], "weights":[.5, .5]}
        with tempfile.TemporaryDirectory() as directory, patch.object(training, "ARTIFACTS", Path(directory)):
            for candidate in [method, blend]:
                write_json(Path(directory)/"study/selection_lock.json", {"selected":{"sp500":[candidate]}, "seeds":[0, 1]})
                training.require_locked_holdout({**method, "seed":1})
                training.require_locked_holdout({**alpha, "seed":1})
                with self.assertRaises(ValueError):
                    training.require_locked_holdout({**alpha, "seed":2})
                with self.assertRaises(ValueError):
                    training.require_locked_holdout({**method, "seed":1, "risk_penalty":1})
        self.assertEqual(frozen_alpha_config({**method, "seed":1})["seed"], 1)

    def test_frozen_checkpoint_and_validation_price_cache_tampering_fail(self):
        config = {"market":"sp500", "seed":0, "family":"historical_risk", "risk_penalty":.5,
                  "alpha_source":{"market":"sp500", "seed":0, "family":"residual"}}
        with tempfile.TemporaryDirectory() as directory, patch.object(historical, "ARTIFACTS", Path(directory)):
            source = frozen_alpha_config(config)
            alpha = Path(directory)/"trials"/config_id(source)
            alpha.mkdir(parents=True)
            (alpha/"best.pt").write_bytes(b"actualcheckpoint")
            cache = Path(directory)/"cache"
            cache.mkdir()
            (cache/"prices.pkl").write_bytes(b"actualprices")
            write_json(cache/"manifest.json", {"files":{"prices.pkl":digest_file(cache/"prices.pkl")}})
            trained = Path(directory)/"trained"
            record = {"family":"historical_risk", "alpha":{"id":config_id(source), "config":source,
                      "model_sha256":{"best.pt":digest_file(alpha/"best.pt")}}, "risk_spec":risk_spec(config),
                      "validation_price_cache":{"path":str(cache), "manifest_sha256":digest_file(cache/"manifest.json"),
                      "files":{"prices.pkl":digest_file(cache/"prices.pkl")}}}
            write_json(trained/"components.json", record)
            self.assertEqual(verify_historical_source(config, trained), record)
            (alpha/"best.pt").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                verify_historical_source(config, trained)
            (alpha/"best.pt").write_bytes(b"actualcheckpoint")
            (cache/"prices.pkl").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                verify_historical_source(config, trained)


if __name__ == "__main__":
    unittest.main()
