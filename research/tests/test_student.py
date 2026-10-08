"""Student likelihood values/gradients, moment semantics, and inference boundaries."""
import copy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.stats import t
import torch

from research.common import write_json
from research.diagnostics import diagnose
from research.models import can_pack_training_days, loss_by_day, make_model, rank_loss, uses_temporal_data
from research.risk_regime import frozen_alpha_config
from research.student import student_cdf, student_log_prob, student_nll, student_quantiles
from research.train import neural_predict


class StudentTests(unittest.TestCase):
    def test_density_and_all_gradients_match_torch_student_with_true_variance(self):
        for df in [3.0, 5.0, 10.0]:
            mean = torch.tensor([-1.0, .2, 3.0, 0.0], dtype=torch.float64, requires_grad=True)
            log_variance = torch.tensor([-6.0, -.3, 1.5, 4.0], dtype=torch.float64, requires_grad=True)
            target = torch.tensor([-.9, -.2, 4.0, 8.0], dtype=torch.float64, requires_grad=True)
            distribution = torch.distributions.StudentT(df, mean, (log_variance.exp() * (df-2) / df).sqrt())
            actual = student_log_prob(mean, log_variance, target, df)
            expected = distribution.log_prob(target)
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
            for first, second in zip(torch.autograd.grad(actual.sum(), [mean, log_variance, target]),
                                     torch.autograd.grad(expected.sum(), [mean, log_variance, target])):
                torch.testing.assert_close(first, second, rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(distribution.variance, log_variance.exp(), rtol=1e-12, atol=1e-12)

    def test_high_df_reduces_to_normal_density_and_gradients(self):
        mean = torch.tensor([-.1, .3, 1.0], dtype=torch.float64, requires_grad=True)
        log_variance = torch.tensor([-.2, .3, 1.0], dtype=torch.float64, requires_grad=True)
        target = torch.tensor([.2, .5, 1.7], dtype=torch.float64)
        actual = student_nll(mean, log_variance, target, 1e6)
        expected = -torch.distributions.Normal(mean, (log_variance/2).exp()).log_prob(target).mean()
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
        for first, second in zip(torch.autograd.grad(actual, [mean, log_variance]),
                                 torch.autograd.grad(expected, [mean, log_variance])):
            torch.testing.assert_close(first, second, rtol=6e-6, atol=2e-6)

    def test_fixed_variance_tail_loss_has_finite_gradients_and_lower_outlier_influence(self):
        mean = torch.zeros(4, dtype=torch.float64, requires_grad=True)
        variance = torch.tensor([-6.0, -6.0, 4.0, 4.0], dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([1e6, -1e6, 1e6, -1e6], dtype=torch.float64)
        loss = student_nll(mean, variance, labels, 3.0)
        self.assertTrue(torch.isfinite(loss))
        gradients = torch.autograd.grad(loss, [mean, variance])
        self.assertTrue(all(torch.isfinite(value).all() for value in gradients))
        gaussian_gradient = (mean.detach()-labels)*(-variance.detach()).exp()/4
        self.assertTrue((gradients[0].abs() < gaussian_gradient.abs()).all())

    def test_quantiles_invert_cdf_and_transform_with_location_and_variance_units(self):
        mean = torch.tensor([-2.0, .3, 4.0], dtype=torch.float64)
        log_variance = torch.tensor([-6.0, .5, 4.0], dtype=torch.float64)
        levels = [.1, .5, .9]
        for df in [3.0, 5.0, 10.0]:
            quantiles = student_quantiles(mean, log_variance, df, levels)
            values = student_cdf(mean[:, None], log_variance[:, None], quantiles, df)
            np.testing.assert_allclose(values, np.tile(levels, (3, 1)), rtol=1e-10, atol=1e-10)
            transformed = student_quantiles(mean*2+7, log_variance+2*math.log(2), df)
            torch.testing.assert_close(transformed, quantiles*2+7, rtol=1e-12, atol=1e-12)
        normal_lower = mean-1.2815515655446004*(log_variance/2).exp()
        self.assertGreater(float((student_quantiles(mean, log_variance, 3)[:, 0]-normal_lower).abs().max()), .1)

    def test_packed_daily_student_losses_and_gradients_keep_ranking_separate(self):
        torch.manual_seed(3)
        config = {"family":"student_t", "student_df":5.0, "width":16, "depth":1, "dropout":0.0}
        model = make_model(config)
        reference = copy.deepcopy(model)
        stock, context, labels = torch.randn(13, 1, 158), torch.randn(13, 76), torch.randn(13)
        labels[5:] += 3
        actual = loss_by_day(model(stock, context), labels, [5, 8], "student_nll_rank", [.5, 1.5])
        expected = torch.stack([rank_loss(reference(stock[:5], context[:5]), labels[:5], "student_nll_rank")*.5,
                                rank_loss(reference(stock[5:], context[5:]), labels[5:], "student_nll_rank")*1.5])
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        actual.mean().backward()
        expected.mean().backward()
        for first, second in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(first.grad, second.grad, atol=2e-6, rtol=2e-5)
        self.assertTrue(can_pack_training_days(config))
        self.assertFalse(can_pack_training_days({**config, "cs_norm":True}))
        self.assertTrue(uses_temporal_data({**config, "encoder":"temporal_mixer"}))

    def test_gaussian_control_initialization_exact_and_prediction_never_loads_labels(self):
        base = {"family":"risk_aware", "width":16, "depth":1, "dropout":0.0, "risk_penalty":.1}
        torch.manual_seed(11)
        gaussian = make_model(base).eval()
        rng = torch.get_rng_state().clone()
        torch.manual_seed(11)
        student = make_model({**base, "family":"student_t", "student_df":3.0}).eval()
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        for name, parameter in gaussian.state_dict().items():
            torch.testing.assert_close(parameter, student.state_dict()[name], rtol=0, atol=0)
        class FeaturesOnly:
            device = torch.device("cpu")
            day_ids = [0, 1]
            @property
            def labels(self):
                raise AssertionError("Student prediction accessed labels")
            def inputs(self, day):
                return torch.ones(4, 1, 158)*day, torch.ones(4, 76)
        np.testing.assert_array_equal(neural_predict(gaussian, FeaturesOnly(), False),
                                      neural_predict(student, FeaturesOnly(), False))
        source = {**base, "family":"student_t", "market":"csi300", "seed":0}
        self.assertEqual(frozen_alpha_config({"market":"csi300", "seed":2, "alpha_source":source})["seed"], 2)

    def test_invalid_df_shapes_and_probabilities_fail(self):
        for value in [None, True, False, 0, 1, 2, -1, float("nan"), float("inf"), "3"]:
            with self.assertRaises(ValueError):
                make_model({"family":"student_t", "student_df":value, "width":16})
        mean, variance = torch.zeros(3), torch.zeros(3)
        for bad in [torch.zeros(2), torch.zeros(3, 1)]:
            with self.assertRaises(ValueError):
                student_nll(mean, variance, bad, 3)
        for bad in [torch.ones(3)*2, torch.tensor(float("nan")), torch.ones(4)*3]:
            with self.assertRaises(ValueError):
                student_nll(mean, variance, mean, bad)
        for levels in [[], [0], [1], [float("nan")]]:
            with self.assertRaises(ValueError):
                student_quantiles(mean, variance, 3, levels)

    def test_diagnostics_use_student_intervals_and_raw_density_without_label_fitting(self):
        config = {"family":"student_t", "market":"sp500", "student_df":3.0,
                  "width":16, "depth":1, "dropout":0.0}
        model = make_model(config).eval()
        stock, context = torch.randn(6, 1, 158), torch.randn(6, 76)
        index = pd.MultiIndex.from_product([pd.to_datetime(["2021-01-04", "2021-01-05"]), ["A", "B", "C"]],
                                          names=["datetime", "instrument"])
        class FeatureOnlyData:
            day_ids = [0, 1]
            def __init__(self, *args, **kwargs):
                if args[1] != "valid":
                    raise AssertionError("Student diagnostics accessed another split")
                self.index = index
            def inputs(self, day):
                return stock[day*3:(day+1)*3], context[day*3:(day+1)*3]
            def selected_positions(self):
                return np.arange(6)
            def batch(self, *args):
                raise AssertionError("Student prediction tried to load labels")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root/"config.json", config)
            write_json(root/"target_transform.json", {"scale":.02, "kind":"raw_standardized"})
            torch.save({"model":model.state_dict(), "epoch":0}, root/"best.pt")
            with patch("research.diagnostics.DailyData", FeatureOnlyData), patch("research.diagnostics.raw_labels", return_value=pd.Series(np.arange(6)*.01, index=index)):
                report = diagnose(root, root/"original")
            with patch("research.diagnostics.DailyData", FeatureOnlyData), patch("research.diagnostics.raw_labels", return_value=pd.Series(np.arange(6)+10, index=index)):
                diagnose(root, root/"poisoned")
            original = pd.read_pickle(root/"original/validation_forecasts.pkl")
            poisoned = pd.read_pickle(root/"poisoned/validation_forecasts.pkl")
            for column in ["score", "mean", "sigma", "q10", "q50", "q90", "lower", "upper"]:
                np.testing.assert_array_equal(original[column], poisoned[column])
            true_scale = original.sigma.to_numpy()*math.sqrt(1/3)
            np.testing.assert_allclose(original.q10, t.ppf(.1, 3, loc=original["mean"], scale=true_scale), rtol=1e-6)
            np.testing.assert_allclose(original.negative_log_density, -t.logpdf(original.target, 3, loc=original["mean"], scale=true_scale), rtol=1e-12)
            self.assertEqual(sum(report["student_t"]["pit_histogram"]), 6)


if __name__ == "__main__":
    unittest.main()
