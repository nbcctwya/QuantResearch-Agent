"""Independent EMA arithmetic, immutable fitted buffers, and exact state continuation."""
import copy
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from research.averaging import ema_decay, make_ema
from research.models import make_model, rank_loss
from research.train import neural_predict, run_training


class AveragingTests(unittest.TestCase):
    def test_configuration_rejects_invalid_decay_and_unoptimized_routes(self):
        for value in [True, False, 0, 1, -0.2, 1.01, float("nan"), float("inf"), "0.99"]:
            with self.assertRaises(ValueError):
                ema_decay({"family": "residual", "ema_decay": value})
        self.assertIsNone(ema_decay({"family": "residual"}))
        for family in ["ridge", "lgbm", "scores_blend"]:
            with tempfile.TemporaryDirectory() as temporary:
                with self.assertRaisesRegex(ValueError, "optimized neural"):
                    run_training({"family": family, "seed": 0, "ema_decay": 0.99}, Path(temporary) / "trial")
        with self.assertRaises(ValueError):
            ema_decay({"family": "risk_overlay", "alpha_source": {"family": "residual"}, "ema_decay": .99})

    def test_exact_manual_double_precision_recursion_and_buffer_copy(self):
        model = nn.Linear(1, 1, bias=False).double()
        model.register_buffer("edges", torch.tensor([1., 2.], dtype=torch.float64))
        model.register_buffer("count", torch.tensor(7))
        averaged = make_ema(model, {"family": "residual", "ema_decay": .75})
        expected = None
        for index, value in enumerate([.2, .6, -.2, 1.8]):
            with torch.no_grad():
                model.weight.fill_(value)
                model.edges.fill_(index + 10.)
                model.count.fill_(index + 50)
            averaged.update_parameters(model)
            expected = value if expected is None else .75 * expected + .25 * value
            torch.testing.assert_close(averaged.module.weight, torch.tensor([[expected]], dtype=torch.float64), atol=1e-15, rtol=0)
            torch.testing.assert_close(averaged.module.edges, model.edges, atol=0, rtol=0)
            self.assertEqual(int(averaged.module.count), int(model.count))
            self.assertEqual(int(averaged.n_averaged), index + 1)
        self.assertTrue(model.weight.requires_grad)
        self.assertFalse(averaged.module.weight.requires_grad)

    def test_disabled_averaging_preserves_rng_and_model(self):
        torch.manual_seed(17)
        model = nn.Linear(5, 2)
        before = torch.get_rng_state().clone()
        state = copy.deepcopy(model.state_dict())
        self.assertIsNone(make_ema(model, {"family": "residual"}))
        torch.testing.assert_close(torch.get_rng_state(), before, atol=0, rtol=0)
        for name, value in state.items():
            torch.testing.assert_close(value, model.state_dict()[name], atol=0, rtol=0)

    def test_ema_does_not_change_raw_training_gradients_or_updates(self):
        torch.manual_seed(3)
        config = {"family": "mixture_gaussian", "width": 16, "depth": 1, "dropout": 0.,
                  "context": True, "mixture_components": 2, "ema_decay": .99}
        model = make_model(config)
        reference = copy.deepcopy(model)
        random_before = torch.get_rng_state().clone()
        averaged = make_ema(model, config)
        torch.testing.assert_close(torch.get_rng_state(), random_before, atol=0, rtol=0)
        stock, context, label = torch.randn(13, 1, 158), torch.randn(13, 76), torch.randn(13)
        optimizers = [torch.optim.AdamW(m.parameters(), lr=.001) for m in [model, reference]]
        for current, optimizer in zip([model, reference], optimizers):
            rank_loss(current(stock, context), label, "gaussian_nll_rank").backward()
            optimizer.step()
        averaged.update_parameters(model)
        for parameter, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(parameter, expected, atol=0, rtol=0)
            torch.testing.assert_close(parameter.grad, expected.grad, atol=0, rtol=0)

    def test_serialized_raw_optimizer_and_ema_continue_exactly(self):
        torch.manual_seed(9)
        config = {"family": "residual", "ema_decay": .99}
        model = nn.Linear(3, 2)
        averaged = make_ema(model, config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        batches = [(torch.randn(7, 3), torch.randn(7, 2)) for _ in range(6)]
        def update(m, o, a, batch):
            o.zero_grad()
            (m(batch[0]) - batch[1]).square().mean().backward()
            o.step()
            a.update_parameters(m)
        for batch in batches[:3]:
            update(model, optimizer, averaged, batch)
        buffer = io.BytesIO()
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "ema": averaged.state_dict()}, buffer)
        buffer.seek(0)
        saved = torch.load(buffer, weights_only=False)
        resumed = nn.Linear(3, 2)
        resumed.load_state_dict(saved["model"])
        resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=.01)
        resumed_optimizer.load_state_dict(saved["optimizer"])
        resumed_ema = make_ema(resumed, config)
        resumed_ema.load_state_dict(saved["ema"])
        for batch in batches[3:]:
            update(model, optimizer, averaged, batch)
            update(resumed, resumed_optimizer, resumed_ema, batch)
        for key, value in averaged.state_dict().items():
            torch.testing.assert_close(value, resumed_ema.state_dict()[key], atol=0, rtol=0)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, resumed.state_dict()[key], atol=0, rtol=0)

    def test_prediction_uses_only_features_and_bn_requires_explicit_protocol(self):
        with self.assertRaisesRegex(ValueError, "BatchNorm"):
            make_ema(nn.BatchNorm1d(2), {"family": "residual", "ema_decay": .99})
        config = {"family": "residual", "width": 16, "depth": 1, "context": True, "dropout": 0., "ema_decay": .99}
        model = make_model(config)
        averaged = make_ema(model, config)
        averaged.update_parameters(model)
        class FeaturesOnly:
            device = torch.device("cpu")
            day_ids = [0, 1]
            @property
            def labels(self):
                raise AssertionError("prediction must never read labels")
            def inputs(self, day):
                return torch.ones(4, 1, 158) * day, torch.ones(4, 76)
        before = int(averaged.n_averaged)
        self.assertEqual(neural_predict(averaged.module, FeaturesOnly(), False).shape, (8,))
        self.assertEqual(int(averaged.n_averaged), before)


if __name__ == "__main__":
    unittest.main()
