"""Ensure packing cannot mix the ranking targets or cross-stock information of different dates."""
import copy
import unittest

import torch

from research.models import make_model,can_pack_training_days,loss_by_day,rank_loss
from research.train import training_group_losses


class BatchingTests(unittest.TestCase):
    def test_packed_daily_losses_and_gradients_match_separate_dates(self):
        torch.manual_seed(27)
        config={"family":"residual","width":32,"depth":1,"dropout":0.,"context":True,"market_gate":True}
        model=make_model(config)
        reference=copy.deepcopy(model)
        counts=[5,9]
        stock,context,labels=torch.randn(14,1,158),torch.randn(14,76),torch.randn(14)
        # A date-level shift makes a wrongly pooled correlation loss observably different.
        labels[5:]+=3.
        prediction=model(stock,context)
        actual=loss_by_day(prediction,labels,counts,"mixed",[0.5,1.5])
        expected=torch.stack([rank_loss(reference(stock[:5],context[:5]),labels[:5],"mixed")*0.5,
                              rank_loss(reference(stock[5:],context[5:]),labels[5:],"mixed")*1.5])
        torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-5)
        actual.mean().backward()
        expected.mean().backward()
        for first,second in zip(model.parameters(),reference.parameters()):
            torch.testing.assert_close(first.grad,second.grad,atol=1e-5,rtol=1e-5)
        pooled=rank_loss(prediction,labels,"mixed")
        self.assertGreater(abs(float(pooled.detach()-actual.mean().detach())),0.01)

    def test_distribution_heads_are_sliced_at_date_boundaries(self):
        torch.manual_seed(4)
        for family,objective in [("risk_aware","gaussian_nll_rank"),("quantile_aware","quantile_rank")]:
            model=make_model({"family":family,"width":32,"depth":1,"dropout":0.,"context":True})
            prediction=model(torch.randn(13,1,158),torch.randn(13,76))
            labels=torch.randn(13)
            actual=loss_by_day(prediction,labels,[4,9],objective)
            self.assertEqual(actual.shape,(2,))
            self.assertTrue(torch.isfinite(actual).all())
            actual.mean().backward()
            self.assertTrue(torch.isfinite(model.output[-1].weight.grad).all())

    def test_cross_stock_models_are_never_packed_across_dates(self):
        self.assertTrue(can_pack_training_days({"family":"residual"}))
        self.assertTrue(can_pack_training_days({"family":"temporal_mixer"}))
        for config in [{"family":"master_control"},{"family":"residual","cs_norm":True},
                       {"family":"temporal_mixer","latent_factors":16},{"family":"batch_ensemble","cs_norm":True}]:
            self.assertFalse(can_pack_training_days(config))

    def test_gradient_accumulation_preserves_independent_cross_stock_attention(self):
        import numpy as np
        torch.manual_seed(35)
        config={"family":"temporal_mixer","width":32,"depth":1,"dropout":0.,"latent_factors":4}
        model=make_model(config)
        reference=copy.deepcopy(model)
        batches=[(torch.randn(count,8,158),torch.randn(count,76),torch.randn(count)) for count in [5,9]]
        class Data:
            device=torch.device("cpu")
            def batch(self,day):
                return batches[day]
        actual=training_group_losses(model,Data(),np.array([0,1]),config,np.array([0.5,1.5]),amp=False)
        expected=torch.stack([rank_loss(reference(stock,context),labels,"mse")*weight
                              for (stock,context,labels),weight in zip(batches,[0.5,1.5])])
        torch.testing.assert_close(actual,expected)
        actual.mean().backward()
        expected.mean().backward()
        for first,second in zip(model.parameters(),reference.parameters()):
            torch.testing.assert_close(first.grad,second.grad)


if __name__=="__main__":
    unittest.main()
