"""Numerical encodings must use only selected training features and survive checkpoint reload."""
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from research.models import make_model
from research.numerical import PiecewiseLinearEncoder,PeriodicFeatureEncoder,fit_model_feature_encoders


class NumericalTests(unittest.TestCase):
    def test_piecewise_ramps_are_continuous_and_extrapolate_at_outer_bins(self):
        encoder = PiecewiseLinearEncoder(1,bins=2,include_raw=False,tail_clip=None)
        encoder.fit(np.array([[0.],[1.],[2.]]))
        x = torch.tensor([[-1.],[0.],[0.5],[1.],[1.5],[2.],[3.]],requires_grad=True)
        expected = torch.tensor([[-1.,0.],[0.,0.],[0.5,0.],[1.,0.],[1.,0.5],[1.,1.],[1.,2.]])
        torch.testing.assert_close(encoder(x),expected)
        encoder(x).sum().backward()
        torch.testing.assert_close(x.grad[[0,2,4,6]],torch.ones(4,1))
        around_edge = encoder(torch.tensor([[1.-1e-5],[1.+1e-5]]))
        self.assertLess(float((around_edge[0]-around_edge[1]).abs().max()),3e-5)

    def test_repeated_quantiles_constant_features_and_checkpoint_reload(self):
        encoder = PiecewiseLinearEncoder(3,bins=8,include_raw=False)
        features = np.array([[0.,5.,0.]]*9+[[1.,5.,2.]],dtype=np.float32)
        metadata = encoder.fit(features)
        self.assertEqual(metadata["constant_features"],1)
        x = torch.tensor([[100.,100.,-100.],[0.5,5.,1.]])
        result = encoder(x)
        self.assertTrue(torch.isfinite(result).all())
        self.assertLessEqual(float(result.max()),3.)
        self.assertGreaterEqual(float(result.min()),-2.)
        torch.testing.assert_close(result[:,8:16],torch.zeros(2,8))
        restored = PiecewiseLinearEncoder(3,bins=8,include_raw=False)
        with self.assertRaisesRegex(ValueError,"must be fitted"):
            restored(x)
        buffer = io.BytesIO()
        torch.save(encoder.state_dict(),buffer)
        buffer.seek(0)
        restored.load_state_dict(torch.load(buffer,weights_only=True))
        self.assertTrue(restored.is_fitted)
        torch.testing.assert_close(restored(x),result,rtol=0,atol=0)

    def test_fit_uses_only_selected_training_features_and_never_labels(self):
        class Training:
            split = "train"
            day_ids = np.array([0,1])
            boundaries = np.array([0,4,8,12])
            dates = np.array(["2020-12-09","2020-12-10","2020-12-11"],dtype="datetime64[D]")
            features = np.broadcast_to(np.arange(12,dtype=np.float32)[:,None],(12,234)).copy()
            features[8:] = 1e9  # Excluded rows must not change the fitted bins.

            def selected_positions(self):
                return np.arange(8,dtype=np.int64)

            @property
            def labels(self):
                raise AssertionError("Labels must not enter a feature-bin fit")

        config = {"family":"residual","feature_encoder":"ple","ple_bins":4,"width":16,"depth":1}
        with tempfile.TemporaryDirectory() as directory:
            model = make_model(config)
            fit_model_feature_encoders(model,Training(),config,Path(directory))
            self.assertLessEqual(float(model.feature_encoder.left.max()),7.)
            metadata = json.loads((Path(directory)/"feature_encoder.json").read_text())
            self.assertEqual(metadata["fit_samples"],8)
            self.assertEqual(metadata["last_training_date"],"2020-12-10")
            self.assertFalse(metadata["labels_used"])
            self.assertFalse(metadata["validation_or_test_features_used"])
            invalid = Training()
            invalid.split = "valid"
            with self.assertRaisesRegex(ValueError,"training split"):
                fit_model_feature_encoders(make_model(config),invalid,config,Path(directory))

    def test_periodic_features_and_distribution_heads_receive_gradients(self):
        torch.manual_seed(3)
        for family in ["residual","batch_ensemble","risk_aware","quantile_aware"]:
            config = {"family":family,"feature_encoder":"periodic","width":32,"depth":1,"dropout":0.}
            model = make_model(config)
            prediction = model(torch.randn(7,1,158),torch.randn(7,76))
            loss = sum(value.square().mean() for value in prediction.values()) if isinstance(prediction,dict) else prediction.square().mean()
            loss.backward()
            encoder = model.feature_encoder
            self.assertIsInstance(encoder,PeriodicFeatureEncoder)
            self.assertTrue(torch.isfinite(encoder.frequencies.grad).all())
            self.assertGreater(float(encoder.frequencies.grad.abs().sum()),0.)


if __name__ == "__main__":
    unittest.main()
