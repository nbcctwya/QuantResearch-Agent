"""Native alpha units, one-sided risk and preserved historical-score behavior."""
import unittest

import numpy as np
import pandas as pd

from research.historical_risk import apply_historical_risk, risk_spec, risk_z
from research.risk_shaping import shaped_risk_scores
from research.signals import normalized_scores


class RiskShapingTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.MultiIndex.from_product([pd.to_datetime(["2021-01-04", "2022-01-04"]),
                                               ["A", "B", "C", "D"]], names=["datetime", "instrument"])
        self.frame = pd.DataFrame({"score":np.array([1, 4, 2, 3, 15, 9, 8, 11], dtype=np.float32),
                                   "label":np.linspace(-.1, .1, 8)}, index=self.index)
        self.features = pd.DataFrame({"beta":[.1, 3, .5, 1, 2, .4, np.nan, .9]}, index=self.index)
        self.z = risk_z(self.features, risk_spec({}))

    def test_native_scale_matches_independent_formula_and_single_seed_ranks(self):
        scores = self.frame.score.astype(np.float64)
        scale = scores.groupby(level="datetime").transform("std", ddof=0).clip(lower=1e-6)
        for transform in ["linear", "positive"]:
            risk = self.z.clip(lower=0) if transform == "positive" else self.z
            actual = shaped_risk_scores(self.frame, self.z, .5, "native", transform)
            np.testing.assert_allclose(actual, scores-.5*scale*risk, rtol=0, atol=1e-15)
            normalized = shaped_risk_scores(self.frame, self.z, .5, "cs_z", transform)
            np.testing.assert_array_equal(actual.groupby(level="datetime").rank(), normalized.groupby(level="datetime").rank())
            mu = scores.groupby(level="datetime").transform("mean")
            np.testing.assert_allclose(actual, mu+scale*normalized, rtol=1e-14, atol=1e-14)

    def test_positive_penalty_never_rewards_low_risk_and_missing_risk_is_neutral(self):
        actual = shaped_risk_scores(self.frame, self.z, 1., "native", "positive")
        np.testing.assert_array_equal(actual[self.z <= 0], self.frame.score[self.z <= 0])
        self.assertTrue((actual[self.z > 0] < self.frame.score[self.z > 0]).all())
        self.assertEqual(actual.loc[(pd.Timestamp("2022-01-04"), "C")], self.frame.score.loc[(pd.Timestamp("2022-01-04"), "C")])
        self.assertTrue((actual <= self.frame.score).all())

    def test_native_scale_preserves_seed_weighting_during_actual_avg_none(self):
        scaled = self.frame.copy()
        scaled.score = 10*self.frame.score+7
        native = [shaped_risk_scores(frame, self.z, .5, "native", "linear") for frame in [self.frame, scaled]]
        standardized = [shaped_risk_scores(frame, self.z, .5, "cs_z", "linear") for frame in [self.frame, scaled]]
        np.testing.assert_allclose(native[1], 10*native[0]+7, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(standardized[0], standardized[1], rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(np.mean(native, axis=0), 5.5*native[0]+3.5, rtol=1e-14, atol=1e-14)

    def test_labels_and_future_dates_cannot_change_earlier_scores(self):
        actual = apply_historical_risk(self.frame, self.features, {"risk_penalty":.5, "alpha_norm":"native", "risk_score_transform":"positive"})
        poisoned = self.frame.copy()
        poisoned.label = 123456789.
        poisoned.loc[pd.IndexSlice[pd.Timestamp("2022-01-04"), :], "score"] = [987, -456, 0, 199]
        risks = self.features.copy()
        risks.loc[pd.IndexSlice[pd.Timestamp("2022-01-04"), :], "beta"] = [876, 54, np.nan, -5]
        changed = apply_historical_risk(poisoned, risks, {"risk_penalty":.5, "alpha_norm":"native", "risk_score_transform":"positive"})
        pd.testing.assert_series_equal(actual.score.loc[:"2021"], changed.score.loc[:"2021"], check_exact=True)

    def test_zero_controls_retain_original_scores_dtype_and_labels_for_all_variants(self):
        for normalization in ["native", "cs_z"]:
            for transform in ["linear", "positive"]:
                config = {"risk_penalty":0., "alpha_norm":normalization, "risk_score_transform":transform}
                pd.testing.assert_frame_equal(apply_historical_risk(self.frame, self.features, config), self.frame, check_exact=True)
                pd.testing.assert_series_equal(shaped_risk_scores(self.frame, self.z, 0., normalization, transform), self.frame.score, check_exact=True)

    def test_existing_spec_and_old_scoring_route_remain_exact(self):
        spec = risk_spec({"risk_penalty":.5})
        self.assertEqual(spec["version"], 1)
        self.assertNotIn("risk_score_transform", spec)
        expected = self.frame.copy()
        expected.score = normalized_scores(self.frame, "cs_z")-.5*self.z
        pd.testing.assert_frame_equal(apply_historical_risk(self.frame, self.features, {"risk_penalty":.5}), expected, check_exact=True)
        self.assertEqual(risk_spec({"risk_penalty":.5, "alpha_norm":"cs_z", "risk_score_transform":"linear"}), spec)

    def test_constant_single_stock_and_duplicate_or_nonfinite_inputs(self):
        constant = self.frame.assign(score=5.)
        actual = shaped_risk_scores(constant, self.z, .5, "native", "positive")
        np.testing.assert_allclose(actual, 5.-.5e-6*self.z.clip(lower=0), rtol=0, atol=0)
        single = self.frame.iloc[:1]
        pd.testing.assert_series_equal(shaped_risk_scores(single, self.z.iloc[:1], .5, "native", "positive"), single.score.astype(float), check_exact=True)
        for frame, risk in [(self.frame.iloc[::-1], self.z.iloc[::-1]),
                            (self.frame.assign(score=np.inf), self.z), (self.frame, self.z+np.nan)]:
            with self.assertRaises(ValueError):
                shaped_risk_scores(frame, risk, .5)

    def test_invalid_new_specs_and_component_spec_changes_are_rejected(self):
        for config in [{"risk_score_transform":"square"}, {"alpha_norm":"none"}, {"alpha_norm":True}]:
            with self.assertRaises(ValueError):
                risk_spec(config)
        for penalty in [-1, True, np.nan, np.inf]:
            with self.assertRaises(ValueError):
                shaped_risk_scores(self.frame, self.z, penalty)
        self.assertNotEqual(risk_spec({"alpha_norm":"native"}), risk_spec({}))
        self.assertNotEqual(risk_spec({"risk_score_transform":"positive"}), risk_spec({}))


if __name__ == "__main__":
    unittest.main()
