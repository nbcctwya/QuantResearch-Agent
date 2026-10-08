"""Protect training-only risk statistics, frozen buffers and static controls."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from research.common import config_id, digest_file, write_json
from research.data import DailyData
from research.models import make_model, rank_loss, prediction_scores
from research.risk_regime import (RiskRegimeScaler, daily_risk_statistic,
                                  fit_frozen_risk_regime, frozen_alpha_config, record_risk_regime_diagnostics)
from research.train import neural_predict, predict_test


class FeatureOnlyDays:
    split = "train"
    device = torch.device("cpu")
    dates = np.arange("2020-01-01", "2020-01-08", dtype="datetime64[D]")
    day_ids = np.array([1, 3, 5])

    def __init__(self):
        self.visited = []

    @property
    def labels(self):
        raise AssertionError("Risk regime fitting must not read labels")

    def batch(self, day):
        raise AssertionError("Risk regime fitting must use features without loading labels")

    def inputs(self, day):
        self.visited.append(int(day))
        stock = torch.zeros(2, 1, 158)
        stock[:, 0, 0] = float(day)
        return stock, torch.zeros(2, 76)


class FixedVariance(nn.Module):
    def forward(self, stock, context):
        return {"log_variance":stock[:, 0, 0]}


class RiskRegimeTests(unittest.TestCase):
    def test_statistic_uses_arithmetic_mean_variance_and_scaling_is_bounded(self):
        log_variance = torch.tensor([-0.4, 0.2, 0.7])
        expected = log_variance.exp().mean().log()
        torch.testing.assert_close(daily_risk_statistic(log_variance), expected)
        scaler = RiskRegimeScaler(strength=0.5)
        with self.assertRaisesRegex(ValueError, "must be fitted"):
            scaler(log_variance)
        scaler.fit([-2.0, 0.0, 2.0])
        low = scaler(log_variance-100.0)
        high = scaler(log_variance+100.0)
        self.assertAlmostEqual(low.item(), 0.5)
        self.assertAlmostEqual(high.item(), 1.5)
        self.assertLess(low.item(), scaler(log_variance).item())
        scaler.fit([1.0, 1.0, 1.0])
        self.assertAlmostEqual(scaler.scale.item(), 0.1)

    def test_fit_and_validation_diagnostics_never_read_labels_and_respect_selected_dates(self):
        source = {"market":"csi300", "family":"risk_aware", "seed":0}
        model = SimpleNamespace(regime=RiskRegimeScaler(0.5), risk_model=FixedVariance().eval(), risk_penalty=0.1)
        data = FeatureOnlyDays()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root/"trials"/config_id(source)/"best.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"frozen variance checkpoint")
            with patch("research.risk_regime.ARTIFACTS", root):
                fit_frozen_risk_regime(model, data, {"risk_source":source}, root/"output", amp=False)
            self.assertEqual(data.visited, [1, 3, 5])
            np.testing.assert_allclose(np.load(root/"output/training_risk_statistics.npy"), [1.0, 3.0, 5.0], atol=1e-6)
            self.assertAlmostEqual(model.regime.center.item(), 3.0)
            metadata = json.loads((root/"output/risk_regime.json").read_text())
            self.assertFalse(metadata["labels_used"])
            self.assertFalse(metadata["validation_or_test_features_used"])
            forbidden = FeatureOnlyDays()
            forbidden.split = "valid"
            with self.assertRaisesRegex(ValueError, "training dates"):
                fit_frozen_risk_regime(model, forbidden, {"risk_source":source}, root/"bad", amp=False)
            self.assertEqual(forbidden.visited, [])
            record_risk_regime_diagnostics(model, forbidden, root/"output", amp=False)
            self.assertEqual(forbidden.visited, [1, 3, 5])
            summary = json.loads((root/"output/valid/risk_regime/summary.json").read_text())
            self.assertFalse(summary["labels_used"])
            forbidden.split = "test"
            with self.assertRaisesRegex(ValueError, "validation features"):
                record_risk_regime_diagnostics(model, forbidden, root/"bad", amp=False)

    def test_checkpoint_reload_preserves_statistics_without_refitting(self):
        first = RiskRegimeScaler(1.0, temperature=2.0)
        first.fit([-1.0, 0.2, 0.6, 1.0])
        second = RiskRegimeScaler(1.0, temperature=2.0)
        second.load_state_dict(first.state_dict())
        self.assertTrue(second.is_fitted)
        variance = torch.tensor([0.3, 0.8])
        torch.testing.assert_close(second(variance), first(variance))
        data = FeatureOnlyDays()
        model = SimpleNamespace(regime=second)
        with patch.object(second, "fit", side_effect=AssertionError("Checkpoint statistics must not be refitted")):
            fit_frozen_risk_regime(model, data, {}, Path("unused"), amp=False)
        self.assertEqual(data.visited, [])

    def test_static_checkpoint_keys_and_frozen_alpha_training_are_preserved(self):
        source = {"market":"csi300", "family":"risk_aware", "seed":0,
                  "width":32, "depth":1, "context":True, "dropout":0.0}
        config = {"market":"csi300", "family":"risk_overlay", "risk_source":source,
                  "width":32, "depth":1, "context":True, "dropout":0.0, "risk_penalty":0.1}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root/"trials"/config_id(source)/"best.pt"
            checkpoint.parent.mkdir(parents=True)
            torch.save({"config":source, "model":make_model(source).state_dict()}, checkpoint)
            with patch("research.models.ARTIFACTS", root):
                static = make_model(config)
                zero = make_model({**config, "risk_regime_strength":0.0})
                dynamic = make_model({**config, "risk_regime_strength":0.5})
            self.assertEqual(set(zero.state_dict()), set(static.state_dict()))
            zero.load_state_dict(static.state_dict(), strict=True)
            missing = dynamic.load_state_dict(static.state_dict(), strict=False)
            self.assertEqual(set(missing.missing_keys), {"regime.center", "regime.scale", "regime.fitted"})
            stock, context = torch.randn(17, 1, 158), torch.randn(17, 76)
            dynamic.train()
            torch.testing.assert_close(dynamic(stock, context), static(stock, context))
            rank_loss(dynamic(stock, context), torch.linspace(-2, 2, 17), "mixed").backward()
            self.assertFalse(dynamic.risk_model.training)
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in dynamic.risk_model.parameters()))
            self.assertTrue(any(p.grad is not None for p in dynamic.alpha.parameters()))
            static.eval()
            zero.eval()
            torch.testing.assert_close(static(stock, context), zero(stock, context))
            dynamic.eval()
            with self.assertRaisesRegex(ValueError, "must be fitted"):
                dynamic(stock, context)
            log_variance = static.risk_model(stock, context)["log_variance"]
            center = daily_risk_statistic(log_variance).item()
            dynamic.regime.fit([center-1, center, center+1])
            torch.testing.assert_close(dynamic(stock, context), static(stock, context))

    def test_data_inputs_exclude_the_label_array(self):
        data = DailyData.__new__(DailyData)
        data.features = np.arange(3*234, dtype=np.float32).reshape(3, 234)
        data.boundaries = np.array([0, 3])
        data.device = torch.device("cpu")
        data.temporal = False
        data.gpu_features = None
        data.labels = object()  # Slicing this would fail.
        stock, context = data.inputs(0)
        torch.testing.assert_close(stock[:, 0], torch.from_numpy(data.features[:, :158]))
        torch.testing.assert_close(context, torch.from_numpy(data.features[:, 158:234]))

    def test_neural_inference_does_not_load_labels(self):
        class FeatureScore(nn.Module):
            def forward(self, stock, context):
                return stock[:, 0, 0]
        data = FeatureOnlyDays()
        actual = neural_predict(FeatureScore(), data, amp=False)
        np.testing.assert_array_equal(actual, [1.0, 1.0, 3.0, 3.0, 5.0, 5.0])
        self.assertEqual(data.visited, [1, 3, 5])

    def test_frozen_alpha_preserves_each_source_score_and_follows_parent_seed(self):
        for family in ["residual", "batch_ensemble", "quantile_aware"]:
            alpha = {"market":"csi300", "family":family, "seed":0, "width":32,
                     "depth":1, "context":True, "dropout":0.1}
            risk = {"market":"csi300", "family":"risk_aware", "seed":0,
                    "width":32, "depth":1, "context":True}
            config = {"market":"csi300", "family":"risk_overlay", "seed":1,
                      "alpha_source":alpha, "risk_source":risk, "risk_penalty":0.0}
            actual_alpha = frozen_alpha_config(config)
            self.assertEqual(actual_alpha["seed"], 1)
            self.assertEqual(alpha["seed"], 0)
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                original = make_model(actual_alpha).eval()
                for source, model in [(actual_alpha, original), (risk, make_model(risk))]:
                    target = root/"trials"/config_id(source)/"best.pt"
                    target.parent.mkdir(parents=True)
                    torch.save({"config":source, "model":model.state_dict(), "epoch":0}, target)
                with patch("research.models.ARTIFACTS", root):
                    frozen = make_model(config)
                frozen.train()
                self.assertFalse(frozen.alpha.training)
                self.assertFalse(frozen.risk_model.training)
                self.assertTrue(all(not p.requires_grad for p in frozen.parameters()))
                frozen.eval()
                stock, context = torch.randn(17, 1, 158), torch.randn(17, 76)
                torch.testing.assert_close(frozen(stock, context), prediction_scores(original(stock, context)))
                with self.assertRaisesRegex(ValueError, "same-market"):
                    frozen_alpha_config({**config, "market":"sp500"})

    def test_changed_frozen_source_is_rejected_before_test_features_are_loaded(self):
        alpha = {"market":"csi300", "family":"residual", "seed":0}
        risk = {"market":"csi300", "family":"risk_aware", "seed":0}
        config = {"market":"csi300", "family":"risk_overlay", "seed":0,
                  "alpha_source":alpha, "risk_source":risk}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root/"scorer"
            paths = []
            for name, source in [("fixed_alpha", alpha), ("risk_source", risk)]:
                target = root/"trials"/config_id(source)/"best.pt"
                target.parent.mkdir(parents=True)
                target.write_bytes(b"frozen checkpoint")
                write_json(output/(name+".json"), {"config":source, "checkpoint_sha256":digest_file(target)})
                paths.append(target)
            with patch("research.risk_regime.ARTIFACTS", root),patch("research.train.require_locked_holdout"),\
                    patch("research.train.DailyData") as data:
                for name, target in zip(["alpha", "risk"], paths):
                    target.write_bytes(b"changed checkpoint")
                    with self.assertRaisesRegex(ValueError, "Frozen "+name+" checkpoint has changed"):
                        predict_test(config, output, root/"holdout")
                    data.assert_not_called()
                    target.write_bytes(b"frozen checkpoint")

    def test_adaptive_training_pool_excludes_frozen_scoring_methods(self):
        from research.runner import next_candidate
        class FixedRandom:
            def random(self):
                return 0.5
            def choice(self, values):
                return values[0]
        train = {"market":"csi300", "family":"residual", "seed":0, "width":128, "depth":2,
                 "objective":"mixed", "dropout":0.1, "lr":0.0005}
        scorer = {"market":"csi300", "family":"risk_overlay", "seed":0, "alpha_source":train}
        leading = [{"config":scorer}, {"config":train}]
        state = {"completed":[config_id(train)], "failed":[], "pending":[], "proposals":0}
        with tempfile.TemporaryDirectory() as temporary, patch("research.runner.ROOT", Path(temporary)),\
                patch("research.runner.read_results", return_value=[]),\
                patch("research.runner.initial_candidates", return_value=[train]),\
                patch("research.runner.ranked_results", return_value=leading),\
                patch("research.runner.random.Random", return_value=FixedRandom()):
            candidate = next_candidate(state)
        self.assertEqual(candidate["family"], "residual")
        self.assertNotIn("alpha_source", candidate)


if __name__ == "__main__":
    unittest.main()
