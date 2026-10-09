"""Separate causal past exposures, training forward labels and forecast inputs."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import research.market_targets as targets
from research.market_targets import (build_training_targets,forward_market_labels,
                                    market_residual_returns,residual_spec,training_market_data,
                                    validate_training_index)
from research.historical_risk import aligned_risk,rolling_price_risk
from research.train import standardized_return_targets


class MarketTargetTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.date_range("2009-01-05",periods=12,freq="B")
        self.index = pd.MultiIndex.from_product([self.dates[:4],["A","B"]],names=["datetime","instrument"])
        self.raw = np.array([.01,-.03,.04,.02,.05,-.01,-.06,.03],dtype=np.float32)
        self.data = SimpleNamespace(split="train",market="csi300",index=self.index,selected_positions=lambda:np.arange(8),
                                    features=np.arange(8*234,dtype=np.float32).reshape(8,234))
        self.features = pd.DataFrame({"beta":[.5,1,2,np.nan,9,-9,0,1.5],"observations":100.,
                                      "market_return":[.01,.01,-.02,-.02,.03,.03,-.01,-.01]},index=self.index)

    def test_market_labels_use_exact_baseline_forward_horizon(self):
        values = pd.Series(np.arange(10,22,dtype=np.float64)**2,index=self.dates)
        actual = forward_market_labels(values,self.index)
        expected = np.repeat([values.iloc[day+5]/values.iloc[day+1]-1 for day in range(4)],2)
        np.testing.assert_array_equal(actual,expected)
        self.assertFalse(np.array_equal(actual,np.repeat([values.iloc[day+5]/values.iloc[day]-1 for day in range(4)],2)))

    def test_raw_stock_returns_beta_clipping_and_missing_exposure_match_formula(self):
        actual = market_residual_returns(self.raw,self.features.market_return,self.features.beta,.5)
        exposure = np.array([.5,1,2,1,3,-3,0,1.5])
        np.testing.assert_array_equal(actual,self.raw.astype(np.float64)-.5*exposure*self.features.market_return.to_numpy())

    def test_strength_zero_is_exact_legacy_target_without_market_access(self):
        expected,scale = standardized_return_targets(self.raw,self.index,"raw_standardized")
        with patch.object(targets,"training_market_data",side_effect=AssertionError("Unexpected price query")):
            actual,new_scale,proof = build_training_targets(self.data,{"market_residual_strength":0.},self.raw)
        np.testing.assert_array_equal(actual,expected)
        self.assertEqual(scale,new_scale)
        self.assertIsNone(proof["market_data"])
        np.testing.assert_array_equal(market_residual_returns(self.raw,None,None,0),self.raw.astype(np.float64))

    def test_scale_fits_residuals_on_selected_training_rows_and_inputs_stay_unchanged(self):
        before=self.data.features.copy()
        with patch.object(targets,"training_market_data",return_value=(self.features,{"manifest_sha256":"fixed"})):
            actual,scale,proof=build_training_targets(self.data,{"market_residual_strength":1.},self.raw)
        residual=market_residual_returns(self.raw,self.features.market_return,self.features.beta,1.)
        expected,new_scale=standardized_return_targets(residual,self.index,"raw_standardized")
        np.testing.assert_array_equal(actual,expected)
        self.assertEqual(scale,new_scale)
        np.testing.assert_array_equal(self.data.features,before)
        self.assertFalse(proof["test_targets_used"])
        self.assertFalse(proof["forecast_inputs_changed"])

    def test_target_signature_changes_with_real_labels_exposure_or_transform(self):
        with patch.object(targets,"training_market_data",return_value=(self.features,{"manifest_sha256":"fixed"})):
            first=build_training_targets(self.data,{"market_residual_strength":.5},self.raw)[2]
            second=build_training_targets(self.data,{"market_residual_strength":1.},self.raw)[2]
            poisoned=build_training_targets(self.data,{"market_residual_strength":.5},self.raw+1)[2]
        self.assertNotEqual(first["target_signature"],second["target_signature"])
        self.assertNotEqual(first["target_signature"],poisoned["target_signature"])

    def test_future_market_labels_change_targets_but_future_prices_cannot_change_past_beta(self):
        rng=np.random.default_rng(7)
        market=rng.normal(0,.01,110)
        stock=.001+1.7*market+rng.normal(0,.003,110)
        dates=pd.date_range("2009-01-05",periods=110,freq="B")
        prices=pd.DataFrame({"M":100*np.cumprod(1+market),"A":25*np.cumprod(1+stock)},index=dates)
        original=rolling_price_risk(prices,"M",40,20)
        changed=prices.copy();changed.iloc[80:]*=np.arange(2,len(changed)-80+2)[:,None]**2
        future=rolling_price_risk(changed,"M",40,20)
        pd.testing.assert_frame_equal(original["beta"].iloc[:80],future["beta"].iloc[:80],check_exact=True)
        for day in [50,70]:
            returns=prices.to_numpy(dtype=np.float64)[1:]/prices.to_numpy(dtype=np.float64)[:-1]-1
            pairs=returns[day-40:day]
            slope=np.linalg.lstsq(np.c_[np.ones(len(pairs)),pairs[:,0]],pairs[:,1],rcond=None)[0][1]
            self.assertAlmostEqual(original["beta"].iloc[day].A,slope,places=12)
        index=pd.MultiIndex.from_product([dates[77:80],["A"]],names=["datetime","instrument"])
        self.assertFalse(np.array_equal(forward_market_labels(prices.M,index),forward_market_labels(changed.M,index)))

    def test_validation_and_test_requests_fail_before_any_cache_or_price_read(self):
        for split in ["valid","test"]:
            self.data.split=split
            with patch.object(targets,"digest_file",side_effect=AssertionError("Cache was read")):
                with self.assertRaises(ValueError):training_market_data(self.data,{})
                with self.assertRaises(ValueError):build_training_targets(self.data,{},self.raw)

    def test_target_dates_and_forward_endpoints_cannot_cross_training_boundary(self):
        for year in [2008,2021,2023]:
            index=pd.MultiIndex.from_product([pd.to_datetime([f"{year}-01-05"]),["A"]],names=["datetime","instrument"])
            with self.assertRaises(ValueError):validate_training_index(index)
        valid_prices=pd.Series(np.arange(12)+10,index=pd.date_range("2021-01-01",periods=12,freq="B"))
        with self.assertRaises(ValueError):forward_market_labels(valid_prices,self.index)
        short=pd.Series(np.arange(7)+10,index=self.dates[:7])
        with self.assertRaises(ValueError):forward_market_labels(short,self.index)

    def test_missing_benchmark_prices_are_never_filled(self):
        prices=pd.Series(np.arange(12)+10.,index=self.dates)
        prices.iloc[5]=np.nan
        with self.assertRaises(ValueError):forward_market_labels(prices,self.index)

    def test_invalid_specs_shapes_and_nonfinite_stock_labels_fail(self):
        for config in [{"market_residual_strength":True},{"market_residual_strength":np.nan},
                       {"market_residual_strength":-1},{"market_residual_strength":2},
                       {"market_beta_window":True},{"market_beta_window":10},{"market_beta_min_observations":1}]:
            with self.assertRaises(ValueError):residual_spec(config)
        for raw,market,beta in [(self.raw,np.ones(7),np.ones(8)),(self.raw,np.full(8,np.inf),np.ones(8)),
                                (self.raw+np.nan,np.ones(8),np.ones(8))]:
            with self.assertRaises(ValueError):market_residual_returns(raw,market,beta,1.)


if __name__ == "__main__":
    unittest.main()
