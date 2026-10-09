"""Training-only risk scaling, bounded missing values, and exact disabled paths."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import research.volatility_targets as module
from research.volatility_targets import (build_volatility_targets,fit_volatility_statistics,
    normalize_base_returns,training_volatility_data,volatility_spec)
from research.train import standardized_return_targets
from research.market_targets import market_residual_returns


class VolatilityTargetTests(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range("2009-01-05",periods=4,freq="B")
        self.index = pd.MultiIndex.from_product([dates,["A","B"]],names=["datetime","instrument"])
        self.raw = np.array([.01,-.03,.04,.02,.05,-.01,-.06,.03],dtype=np.float32)
        self.data = SimpleNamespace(split="train",market="csi300",index=self.index,
            selected_positions=lambda:np.arange(8),features=np.arange(8*234).reshape(8,234))
        self.features = pd.DataFrame({"total_volatility":[.01,.02,.03,.04,.05,0,np.nan,.06],
            "idiosyncratic_volatility":[.02,.01,.04,.03,.06,np.nan,0,.05]},index=self.index)

    def test_floor_and_missing_replacement_fit_selected_positive_training_sigma(self):
        sigma = np.array([0,np.nan,np.inf,-.01,.01,.02,.03,.04])
        statistics=fit_volatility_statistics(sigma)
        self.assertAlmostEqual(statistics["floor"],.0115,places=15)
        self.assertEqual(statistics["positive_median"],.025)
        self.assertEqual(statistics["zero_rows"],1)
        self.assertEqual(statistics["missing_or_negative_rows"],3)
        self.assertEqual(statistics["positive_finite_rows"],4)

    def test_zero_sigma_uses_floor_and_invalid_sigma_uses_fitted_median(self):
        sigma=np.array([0,np.nan,np.inf,-.01,.01,.02,.03,.04])
        stats=fit_volatility_statistics(sigma)
        expected=np.array([stats["floor"],.025,.025,.025,stats["floor"],.02,.03,.04])
        np.testing.assert_array_equal(normalize_base_returns(np.ones(8),sigma,1,stats),1/expected)

    def test_half_power_and_unit_power_match_independent_formula(self):
        sigma=np.arange(1,9)*.01
        stats=fit_volatility_statistics(sigma)
        bounded=np.maximum(sigma,stats["floor"])
        for power in [.5,1.]:
            actual=normalize_base_returns(self.raw,sigma,power,stats)
            expected=self.raw.astype(np.float64)/(np.sqrt(bounded) if power==.5 else bounded)
            np.testing.assert_allclose(actual,expected,rtol=0,atol=2e-15)

    def test_zero_power_raw_is_exact_legacy_and_never_reads_market_or_volatility(self):
        expected,scale=standardized_return_targets(self.raw,self.index,"raw_standardized")
        with patch.object(module,"training_volatility_data",side_effect=AssertionError("Unexpected past price query")), \
             patch.object(module,"build_training_targets",side_effect=AssertionError("Unexpected market query")):
            actual,new_scale,proof=build_volatility_targets(self.data,{"volatility_target_power":0},self.raw)
        np.testing.assert_array_equal(actual,expected)
        self.assertEqual(scale,new_scale)
        self.assertIsNone(proof["volatility_data"])
        self.assertIsNone(proof["volatility_statistics"])
        np.testing.assert_array_equal(normalize_base_returns(self.raw,None,0,None),self.raw.astype(np.float64))

    def test_zero_power_residual_delegates_exact_legacy_base_without_volatility(self):
        residual=market_residual_returns(self.raw,np.full(8,.01),np.ones(8),1)
        expected,scale=standardized_return_targets(residual,self.index,"raw_standardized")
        base_proof={"target_signature":"original-base"}
        with patch.object(module,"training_volatility_data",side_effect=AssertionError("Unexpected volatility query")), \
             patch.object(module,"build_training_targets",return_value=(expected,scale,base_proof)) as original:
            actual,new_scale,proof=build_volatility_targets(self.data,
                {"volatility_target_base":"market_residual","volatility_target_power":0},self.raw)
        original.assert_called_once()
        np.testing.assert_array_equal(actual,expected)
        self.assertEqual(new_scale,scale)
        self.assertEqual(proof["zero_power_base_signature"],"original-base")

    def test_raw_risk_targets_scale_only_after_selected_row_normalization_and_keep_inputs(self):
        before=self.data.features.copy()
        metadata={"manifest_sha256":"frozen"}
        with patch.object(module,"training_volatility_data",return_value=(self.features,metadata)):
            actual,scale,proof=build_volatility_targets(self.data,{"volatility_target_power":.5},self.raw)
        positive=np.array([.01,.02,.03,.04,.05,.06])
        floor=np.quantile(positive,.05)
        sigma=np.array([floor,.02,.03,.04,.05,floor,np.median(positive),.06])
        expected,new_scale=standardized_return_targets(self.raw.astype(np.float64)/np.sqrt(sigma),self.index,"raw_standardized")
        np.testing.assert_array_equal(actual,expected)
        self.assertEqual(scale,new_scale)
        self.assertFalse(proof["forecast_inputs_changed"])
        self.assertFalse(proof["forward_market_labels_used"])
        np.testing.assert_array_equal(self.data.features,before)

    def test_signature_changes_with_target_power_actual_volatility_and_raw_returns(self):
        with patch.object(module,"training_volatility_data",return_value=(self.features,{"manifest_sha256":"original"})):
            first=build_volatility_targets(self.data,{"volatility_target_power":.5},self.raw)[2]
            second=build_volatility_targets(self.data,{"volatility_target_power":1.},self.raw)[2]
            changed=build_volatility_targets(self.data,{"volatility_target_power":.5},self.raw+1)[2]
        with patch.object(module,"training_volatility_data",return_value=(self.features*2,{"manifest_sha256":"changed"})):
            sigma_changed=build_volatility_targets(self.data,{"volatility_target_power":.5},self.raw)[2]
        self.assertEqual(len({proof["target_signature"] for proof in [first,second,changed,sigma_changed]}),4)

    def test_validation_and_test_requests_fail_before_cache_or_price_access(self):
        for split in ["valid","test"]:
            self.data.split=split
            with patch.object(module,"training_market_data",side_effect=AssertionError("Market data accessed")):
                with self.assertRaises(ValueError):training_volatility_data(self.data,{})
                with self.assertRaises(ValueError):build_volatility_targets(self.data,{},self.raw)

    def test_floors_bound_very_small_risk_and_require_positive_training_evidence(self):
        stats=fit_volatility_statistics(np.array([1e-12,2e-12,0]))
        self.assertEqual(stats["floor"],1e-6)
        actual=normalize_base_returns(np.ones(3),[1e-12,0,np.nan],1,stats)
        np.testing.assert_array_equal(actual,np.full(3,1e6))
        for values in [[],[0,0],[np.nan,np.inf,-1]]:
            with self.assertRaises(ValueError):fit_volatility_statistics(values)

    def test_invalid_specs_shapes_or_nonfinite_base_returns_fail(self):
        for config in [{"volatility_target_base":"other"},{"volatility_target_mode":"beta"},
            {"volatility_target_power":True},{"volatility_target_power":np.inf},
            {"volatility_target_power":-1},{"volatility_target_power":2}]:
            with self.assertRaises(ValueError):volatility_spec(config)
        stats=fit_volatility_statistics([.01,.02])
        for base,sigma in [([1,np.nan],[.01,.02]),([1,2],[.01]),([],[])]:
            with self.assertRaises(ValueError):normalize_base_returns(base,sigma,1,stats)


if __name__ == "__main__":
    unittest.main()
