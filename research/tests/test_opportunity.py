"""Verify feature-only fitting, frozen scoring controls and checkpoint recovery."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from research.common import config_id, write_json
from research.models import make_model, prediction_scores
from research.opportunity import (AlphaOpportunityScaler, daily_alpha_statistic,
                                  fit_frozen_alpha_opportunity, record_alpha_opportunity_diagnostics)


class FeatureOnlyDays:
    split = "train"
    device = torch.device("cpu")
    dates = np.arange("2020-01-01", "2020-01-08", dtype="datetime64[D]")
    day_ids = np.array([1, 3, 5])

    def __init__(self):
        self.visited = []

    @property
    def labels(self):
        raise AssertionError("Opportunity fitting and prediction must not read labels")

    def batch(self, day):
        raise AssertionError("Opportunity statistics must use only features")

    def inputs(self, day):
        self.visited.append(int(day))
        stock = torch.zeros(2, 1, 158)
        stock[:, 0, 0] = float(day)
        return stock, torch.zeros(2, 76)


class FrozenAlpha(nn.Module):
    def forward(self, stock, context):
        return stock[:, 0, 0]*stock.new_tensor([-1., 1.])


class OpportunityTests(unittest.TestCase):
    def test_dispersion_ignores_common_score_offset_and_stock_order(self):
        scores = torch.tensor([-1., 0., 2., 7.])
        statistic = daily_alpha_statistic(scores)
        torch.testing.assert_close(daily_alpha_statistic(scores+100.), statistic)
        torch.testing.assert_close(daily_alpha_statistic(scores[[3, 0, 2, 1]]), statistic)
        self.assertTrue(torch.isfinite(daily_alpha_statistic(torch.ones(9))))
        self.assertTrue(torch.isfinite(daily_alpha_statistic(torch.ones(1))))
        with self.assertRaisesRegex(ValueError, "single-date"):
            daily_alpha_statistic(scores[:, None])
        with self.assertRaisesRegex(ValueError, "single-date"):
            daily_alpha_statistic(torch.empty(0))

    def test_multiplier_is_bounded_and_reduces_penalty_for_more_dispersion(self):
        scaler = AlphaOpportunityScaler(0.5)
        with self.assertRaisesRegex(ValueError, "must be fitted"):
            scaler(torch.tensor([-1., 1.]))
        scaler.fit([-2., 0., 2.])
        low = scaler(torch.tensor([-1e-6, 1e-6]))
        middle = scaler(torch.tensor([-1., 1.]))
        high = scaler(torch.tensor([-1e6, 1e6]))
        self.assertGreater(low.item(), middle.item())
        self.assertGreater(middle.item(), high.item())
        self.assertGreaterEqual(high.item(), 0.5)
        self.assertLessEqual(low.item(), 1.5)
        for strength, temperature in [(-0.1, 1.), (1.1, 1.), (float("nan"), 1.), (0.5, 0.)]:
            with self.assertRaises(ValueError):
                AlphaOpportunityScaler(strength, temperature)

    def test_fit_uses_only_selected_frozen_training_forecasts(self):
        source = {"market":"csi300", "family":"residual", "seed":0}
        config = {"market":"csi300", "seed":0, "alpha_source":source}
        model = SimpleNamespace(opportunity=AlphaOpportunityScaler(0.5), freeze_alpha=True,
                                alpha=FrozenAlpha().eval())
        data = FeatureOnlyDays()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root/"trials"/config_id(source)/"best.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"frozen alpha")
            with patch("research.opportunity.ARTIFACTS", root):
                fit_frozen_alpha_opportunity(model, data, config, root/"output", amp=False)
            self.assertEqual(data.visited, [1, 3, 5])
            np.testing.assert_allclose(np.load(root/"output/training_alpha_statistics.npy"), np.log([1., 3., 5.]), atol=1e-6)
            self.assertAlmostEqual(model.opportunity.center.item(), np.log(3.), places=6)
            metadata = json.loads((root/"output/alpha_opportunity.json").read_text())
            self.assertFalse(metadata["labels_used"])
            self.assertFalse(metadata["validation_or_test_features_used"])
            self.assertEqual(metadata["days"], 3)
            before = (root/"output/alpha_opportunity.json").read_bytes()
            data.visited.clear()
            with patch.object(model.opportunity, "fit", side_effect=AssertionError("No refitting on recovery")):
                fit_frozen_alpha_opportunity(model, data, config, root/"output", amp=False)
            self.assertEqual(data.visited, [])
            self.assertEqual((root/"output/alpha_opportunity.json").read_bytes(), before)

    def test_validation_test_unfrozen_and_dropout_fitting_are_rejected(self):
        data = FeatureOnlyDays()
        model = SimpleNamespace(opportunity=AlphaOpportunityScaler(0.5), freeze_alpha=True,
                                alpha=FrozenAlpha().eval())
        for split in ["valid", "test"]:
            data.split = split
            with self.assertRaisesRegex(ValueError, "training dates"):
                fit_frozen_alpha_opportunity(model, data, {}, Path("unused"), amp=False)
            self.assertEqual(data.visited, [])
        data.split = "train"
        model.freeze_alpha = False
        with self.assertRaisesRegex(ValueError, "frozen alpha"):
            fit_frozen_alpha_opportunity(model, data, {}, Path("unused"), amp=False)
        model.freeze_alpha = True
        model.alpha.train()
        with self.assertRaisesRegex(ValueError, "eval mode"):
            fit_frozen_alpha_opportunity(model, data, {}, Path("unused"), amp=False)

    def test_checkpoint_load_recovers_multiplier_and_skips_training_forecasts(self):
        original = AlphaOpportunityScaler(1., temperature=2.)
        original.fit([-5., -4., -3., -2.])
        loaded = AlphaOpportunityScaler(1., temperature=2.)
        loaded.load_state_dict(original.state_dict())
        self.assertTrue(loaded.is_fitted)
        scores = torch.tensor([-0.1, 0., 0.3])
        torch.testing.assert_close(loaded(scores), original(scores), rtol=0, atol=0)
        data = FeatureOnlyDays()
        model = SimpleNamespace(opportunity=loaded, freeze_alpha=True)
        fit_frozen_alpha_opportunity(model, data, {}, Path("unused"), amp=False)
        self.assertEqual(data.visited, [])

    def test_validation_diagnostics_use_only_features_and_preserve_split(self):
        data = FeatureOnlyDays()
        data.split = "valid"
        opportunity = AlphaOpportunityScaler(0.5)
        opportunity.fit(np.log([1., 3., 5.]))
        model = SimpleNamespace(opportunity=opportunity, alpha=FrozenAlpha().eval(),
                                regime=None, risk_penalty=0.1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record_alpha_opportunity_diagnostics(model, data, root, amp=False)
            self.assertEqual(data.visited, [1, 3, 5])
            summary = json.loads((root/"valid/alpha_opportunity/summary.json").read_text())
            self.assertEqual(summary["days"], 3)
            self.assertFalse(summary["labels_used"])
            data.split = "test"
            data.visited.clear()
            with self.assertRaisesRegex(ValueError, "validation features"):
                record_alpha_opportunity_diagnostics(model, data, root/"forbidden", amp=False)
            self.assertEqual(data.visited, [])

    def test_frozen_overlay_combines_scalers_and_preserves_original_checkpoint(self):
        alpha = {"market":"csi300", "family":"residual", "seed":0, "width":32,
                 "depth":1, "context":True, "dropout":0.0}
        risk = {**alpha, "family":"risk_aware"}
        config = {"market":"csi300", "family":"risk_overlay", "seed":0,
                  "alpha_source":alpha, "risk_source":risk, "risk_penalty":0.1}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for source in [alpha, risk]:
                target = root/"trials"/config_id(source)/"best.pt"
                target.parent.mkdir(parents=True)
                torch.save({"config":source, "model":make_model(source).state_dict(), "epoch":0}, target)
            with patch("research.models.ARTIFACTS", root):
                static = make_model(config).eval()
                zero = make_model({**config, "alpha_opportunity_strength":0.0}).eval()
                dynamic = make_model({**config, "alpha_opportunity_strength":0.5,
                                      "risk_regime_strength":0.5}).eval()
                restored = make_model({**config, "alpha_opportunity_strength":0.5,
                                       "risk_regime_strength":0.5}).eval()
                with self.assertRaisesRegex(ValueError, "frozen alpha"):
                    make_model({k:v for k,v in {**config, "alpha_opportunity_strength":0.5}.items() if k!="alpha_source"})
            self.assertEqual(set(static.state_dict()), set(zero.state_dict()))
            self.assertEqual(set(dynamic.state_dict())-set(static.state_dict()),
                             {"regime.center", "regime.scale", "regime.fitted",
                              "opportunity.center", "opportunity.scale", "opportunity.fitted"})
            dynamic.regime.fit([-1., 0., 1.])
            dynamic.opportunity.fit([-5., -3., -1.])
            restored.load_state_dict(dynamic.state_dict())
            self.assertTrue(restored.opportunity.is_fitted)
            stock, context = torch.randn(17, 1, 158), torch.randn(17, 76)
            torch.testing.assert_close(static(stock, context), zero(stock, context), rtol=0, atol=0)
            alpha_score = prediction_scores(dynamic.alpha(stock, context))
            log_variance = dynamic.risk_model(stock, context)["log_variance"]
            penalty = dynamic.risk_penalty*dynamic.regime(log_variance)*dynamic.opportunity(alpha_score)
            standardized = (log_variance-log_variance.mean())/log_variance.std(correction=0).clamp_min(0.1)
            torch.testing.assert_close(dynamic(stock, context), alpha_score-penalty*standardized)
            torch.testing.assert_close(restored(stock, context), dynamic(stock, context), rtol=0, atol=0)
            self.assertTrue(all(not p.requires_grad for p in dynamic.parameters()))
            dynamic.train()
            self.assertFalse(dynamic.alpha.training)
            self.assertFalse(dynamic.risk_model.training)


if __name__ == "__main__":
    unittest.main()
