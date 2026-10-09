"""Independent robust objective values, gradients, date isolation and legacy probes."""
import importlib.util
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F

from research import ARTIFACTS
from research.models import loss_by_day,rank_loss
from research.robust_loss import robust_regression_loss,robust_loss_spec,training_huber_delta,validate_delta

original=ARTIFACTS/"code_releases/cc45e62adf16652556ba1511/research/models.py"
spec=importlib.util.spec_from_file_location("research.original_models_robust_probe",original)
legacy=importlib.util.module_from_spec(spec);sys.modules[spec.name]=legacy;spec.loader.exec_module(legacy)


class RobustLossTests(unittest.TestCase):
    def test_independent_piecewise_values_and_clipped_gradients(self):
        errors=torch.tensor([-4.,-2.,-.5,0.,.5,2.,4.],dtype=torch.float64,requires_grad=True)
        value=robust_regression_loss(errors,torch.zeros_like(errors),2.)
        self.assertAlmostEqual(value.item(),(12+4+.25+0+.25+4+12)/7)
        gradient,=torch.autograd.grad(value,errors)
        torch.testing.assert_close(gradient,torch.tensor([-4.,-4.,-1.,0.,1.,4.,4.],dtype=torch.float64)/7,rtol=0,atol=0)

    def test_large_threshold_matches_original_square_and_gradient(self):
        torch.manual_seed(617)
        scores=torch.randn(41,dtype=torch.float64,requires_grad=True);labels=torch.randn(41,dtype=torch.float64)
        actual=robust_regression_loss(scores,labels,100.)
        expected=F.mse_loss(scores,labels)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        a,=torch.autograd.grad(actual,scores,retain_graph=True);b,=torch.autograd.grad(expected,scores)
        torch.testing.assert_close(a,b,rtol=1e-14,atol=1e-14)

    def test_disabled_all_legacy_objectives_preserve_loss_gradients_and_rng(self):
        for dtype in [torch.float32,torch.float64]:
            for objective in ["mse","mixed","corr","tail_pair","top30_pair","listwise"]:
                with self.subTest(dtype=dtype,objective=objective):
                    torch.manual_seed(91)
                    scores=torch.randn(81,dtype=dtype,requires_grad=True);labels=torch.randn(81,dtype=dtype)
                    rng=torch.get_rng_state();old=legacy.rank_loss(scores,labels,objective);old_rng=torch.get_rng_state()
                    torch.set_rng_state(rng);new=rank_loss(scores,labels,objective,None)
                    self.assertTrue(torch.equal(torch.get_rng_state(),old_rng))
                    torch.testing.assert_close(old,new,rtol=0,atol=0)
                    a,=torch.autograd.grad(old,scores,retain_graph=True);b,=torch.autograd.grad(new,scores)
                    torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_disabled_packed_day_weights_match_actual_old_module(self):
        torch.manual_seed(19)
        scores=torch.randn(59,dtype=torch.float64,requires_grad=True);labels=torch.randn(59,dtype=torch.float64)
        counts=[19,40];weights=[.4,1.3]
        old=legacy.loss_by_day(scores,labels,counts,"mixed",weights).mean()
        new=loss_by_day(scores,labels,counts,"mixed",weights,None).mean()
        torch.testing.assert_close(old,new,rtol=0,atol=0)
        a,=torch.autograd.grad(old,scores,retain_graph=True);b,=torch.autograd.grad(new,scores)
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_packed_robust_loss_keeps_dates_separate(self):
        torch.manual_seed(12)
        scores=torch.randn(64,dtype=torch.float64,requires_grad=True);labels=torch.randn(64,dtype=torch.float64)
        labels[:27]+=5;counts=[27,37];weights=[.2,1.7]
        packed=loss_by_day(scores,labels,counts,"mixed",weights,.5).mean()
        expected=(.2*rank_loss(scores[:27],labels[:27],"mixed",.5)+1.7*rank_loss(scores[27:],labels[27:],"mixed",.5))/2
        torch.testing.assert_close(packed,expected,rtol=0,atol=0)
        a,=torch.autograd.grad(packed,scores,retain_graph=True);b,=torch.autograd.grad(expected,scores,retain_graph=True)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertGreater(abs((packed-rank_loss(scores,labels,"mixed",.5)).item()),1e-3)

    def test_mixed_regression_replacement_preserves_old_correlation_term(self):
        scores=torch.tensor([-.8,.2,1.,1.5],dtype=torch.float64,requires_grad=True)
        labels=torch.tensor([-4.,-.1,.5,5.],dtype=torch.float64)
        a=scores-scores.mean();b=labels-labels.mean();correlation=(a*b).sum()/(a.norm()*b.norm()).clamp_min(1e-6)
        expected=.5*robust_regression_loss(scores,labels,.5)+.5*(1-correlation)
        actual=rank_loss(scores,labels,"mixed",.5)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    def test_large_error_gradient_is_bounded_and_target_stays_intact(self):
        labels=torch.tensor([1.,100.,-100.],dtype=torch.float64);saved=labels.clone()
        scores=torch.zeros(3,dtype=torch.float64,requires_grad=True)
        value=rank_loss(scores,labels,"mse",.5);gradient,=torch.autograd.grad(value,scores)
        self.assertTrue((gradient.abs()<=1/3).all());self.assertTrue(torch.equal(labels,saved))
        self.assertGreater(F.mse_loss(scores,labels).item(),value.item()*50)

    def test_invalid_thresholds_fail_and_none_disables(self):
        self.assertIsNone(validate_delta(None));self.assertIsNone(robust_loss_spec({"family":"residual"}))
        for delta in [0,-1,True,float("nan"),float("inf"),"1"]:
            with self.subTest(delta=delta),self.assertRaises(ValueError):validate_delta(delta)

    def test_unsupported_distribution_and_ranking_objectives_fail(self):
        for family,objective in [("risk_aware","mixed"),("residual","tail_pair"),("lgbm","mse")]:
            with self.subTest(family=family,objective=objective),self.assertRaises(ValueError):
                training_huber_delta({"family":family,"objective":objective,"huber_delta":1.})
        with self.assertRaises(ValueError):rank_loss({"score":torch.zeros(4)},torch.ones(4),"mixed",1.)
        with self.assertRaises(ValueError):rank_loss(torch.zeros(4),torch.ones(4),"listwise",1.)

    def test_robust_metadata_is_fixed_without_data_fitting(self):
        config={"family":"residual","objective":"mixed","huber_delta":2.,"market":"csi300"}
        proof=robust_loss_spec(config)
        self.assertEqual(proof["huber_delta"],2.)
        self.assertEqual(proof["mixed_weights"],{"regression":.5,"correlation":.5})
        self.assertEqual(proof,robust_loss_spec({**config,"market":"sp500"}))


if __name__=="__main__":unittest.main()
