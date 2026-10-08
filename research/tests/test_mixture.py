"""Mixture math is checked against PyTorch distributions and matched Gaussian controls."""
import copy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from research.common import write_json
from research.diagnostics import diagnose
from research.mixture import (gaussian_mixture_cdf, gaussian_mixture_log_prob,
                              gaussian_mixture_moments, gaussian_mixture_nll, gaussian_mixture_quantiles)
from research.models import make_model, can_pack_training_days, loss_by_day, rank_loss, uses_temporal_data
from research.risk_regime import frozen_alpha_config


def reference_distribution(means,log_variances,logits):
    return torch.distributions.MixtureSameFamily(torch.distributions.Categorical(logits=logits),
                  torch.distributions.Normal(means,torch.exp(0.5*log_variances)))


class MixtureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(931)

    def test_density_moments_cdf_and_all_gradients_match_torch_distribution(self):
        means = torch.randn(13,4,dtype=torch.float64,requires_grad=True)
        log_variances = torch.randn(13,4,dtype=torch.float64,requires_grad=True)
        logits = torch.randn(13,4,dtype=torch.float64,requires_grad=True)
        target = torch.randn(13,dtype=torch.float64)
        reference = reference_distribution(means,log_variances,logits)
        torch.testing.assert_close(gaussian_mixture_log_prob(means,log_variances,logits,target),reference.log_prob(target),rtol=1e-12,atol=1e-12)
        actual = gaussian_mixture_nll(means,log_variances,logits,target)
        expected = -reference.log_prob(target).mean()-0.5*math.log(2*math.pi)
        torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
        actual_grad = torch.autograd.grad(actual,[means,log_variances,logits],retain_graph=True)
        expected_grad = torch.autograd.grad(expected,[means,log_variances,logits])
        for first,second in zip(actual_grad,expected_grad):
            torch.testing.assert_close(first,second,rtol=1e-11,atol=1e-11)
        mean,within,between = gaussian_mixture_moments(means,log_variances,logits)
        torch.testing.assert_close(mean,reference.mean)
        torch.testing.assert_close(within+between,reference.variance)
        torch.testing.assert_close(gaussian_mixture_cdf(means,log_variances,logits,target),reference.cdf(target))

    def test_single_component_likelihood_and_initial_parameters_match_gaussian_control(self):
        config = {"width":32,"depth":1,"dropout":0.,"context":True,"market_gate":True,
                  "risk_exponent":0.5,"risk_penalty":0.1}
        torch.manual_seed(58)
        baseline = make_model({**config,"family":"risk_aware"})
        torch.manual_seed(58)
        mixture = make_model({**config,"family":"mixture_gaussian","mixture_components":1})
        self.assertEqual(set(baseline.state_dict()),set(mixture.state_dict()))
        for key,value in baseline.state_dict().items():
            torch.testing.assert_close(value,mixture.state_dict()[key],rtol=0,atol=0)
        stock,context,target = torch.randn(9,1,158),torch.randn(9,76),torch.randn(9)
        first,second = baseline(stock,context),mixture(stock,context)
        for key in first:
            torch.testing.assert_close(first[key],second[key],atol=2e-7,rtol=1e-6)
        actual = rank_loss(second,target,"gaussian_nll_rank")
        expected = rank_loss(first,target,"gaussian_nll_rank")
        torch.testing.assert_close(actual.float(),expected,atol=1e-6,rtol=1e-6)
        actual_grad = torch.autograd.grad(actual,tuple(mixture.parameters()))
        expected_grad = torch.autograd.grad(expected,tuple(baseline.parameters()))
        for a,b in zip(actual_grad,expected_grad):
            torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)

    def test_true_quantiles_invert_bimodal_cdf_and_reduce_to_normal_icdf(self):
        means = torch.tensor([[-4.,4.],[-1.,2.]],dtype=torch.float64)
        log_variances = torch.tensor([[-3.,-3.],[-1.,0.5]],dtype=torch.float64)
        logits = torch.tensor([[0.,0.],[0.,2.]],dtype=torch.float64)
        levels = (0.1,0.5,0.9)
        actual = gaussian_mixture_quantiles(means,log_variances,logits,levels)
        torch.testing.assert_close(gaussian_mixture_cdf(means,log_variances,logits,actual),torch.tensor(levels,dtype=torch.float64).expand(2,-1),rtol=1e-10,atol=1e-10)
        mean,within,between = gaussian_mixture_moments(means,log_variances,logits)
        normal_lower = mean-1.2815515655446004*(within+between).sqrt()
        self.assertGreater(float((actual[:,0]-normal_lower).abs().max()),0.5)
        one = gaussian_mixture_quantiles(means[:,:1],log_variances[:,:1],logits[:,:1],levels)
        reference = torch.distributions.Normal(means[:,:1],torch.exp(0.5*log_variances[:,:1]))
        torch.testing.assert_close(one,reference.icdf(torch.tensor(levels,dtype=torch.float64)),rtol=1e-12,atol=1e-12)

    def test_component_permutations_logit_shifts_and_extreme_tails_are_stable(self):
        means = torch.tensor([[-3.,4.,0.],[100.,-100.,0.]],dtype=torch.float64,requires_grad=True)
        log_variances = torch.tensor([[-6.,4.,0.],[-6.,-6.,4.]],dtype=torch.float64,requires_grad=True)
        logits = torch.tensor([[1000.,-1000.,0.],[-1000.,1000.,0.]],dtype=torch.float64,requires_grad=True)
        target = torch.tensor([1e5,-1e5],dtype=torch.float64)
        actual = gaussian_mixture_nll(means,log_variances,logits,target)
        perm = [2,0,1]
        torch.testing.assert_close(actual,gaussian_mixture_nll(means[:,perm],log_variances[:,perm],logits[:,perm]+123.,target))
        self.assertTrue(torch.isfinite(actual))
        for gradient in torch.autograd.grad(actual,[means,log_variances,logits]):
            self.assertTrue(torch.isfinite(gradient).all())

    def test_packed_losses_and_gradients_keep_dates_separate_and_train_gate_and_heads(self):
        config = {"family":"mixture_gaussian","mixture_components":4,"mixture_gate_input":"stock_market",
                  "width":32,"depth":1,"dropout":0.,"risk_penalty":0.1}
        model = make_model(config)
        reference = copy.deepcopy(model)
        stock,context,labels = torch.randn(13,1,158),torch.randn(13,76),torch.randn(13)
        labels[5:]+=3.
        actual = loss_by_day(model(stock,context),labels,[5,8],"gaussian_nll_rank",[0.5,1.5])
        expected = torch.stack([rank_loss(reference(stock[:5],context[:5]),labels[:5],"gaussian_nll_rank")*0.5,
                                rank_loss(reference(stock[5:],context[5:]),labels[5:],"gaussian_nll_rank")*1.5])
        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
        actual.mean().backward()
        expected.mean().backward()
        for first,second in zip(model.parameters(),reference.parameters()):
            torch.testing.assert_close(first.grad,second.grad,atol=2e-6,rtol=2e-5)
        for gradient in [model.output[-1].weight.grad[0::2],model.output[-1].weight.grad[1::2],model.mixture_gate[-1].weight.grad]:
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()),0.)
        self.assertTrue(can_pack_training_days(config))
        self.assertFalse(can_pack_training_days({**config,"cs_norm":True}))
        self.assertFalse(can_pack_training_days({**config,"latent_factors":4}))
        self.assertTrue(uses_temporal_data({**config,"encoder":"temporal_mixer"}))
        output = model(stock,context)
        torch.testing.assert_close(output["log_variance"].exp(),output["within_variance"]+output["between_variance"])

    def test_market_gate_uses_market_features_and_never_jkp_or_labels(self):
        config = {"family":"mixture_gaussian","mixture_components":3,"width":32,"depth":1,"dropout":0.}
        model = make_model(config).eval()
        with torch.no_grad():
            model.mixture_gate[-1].weight.normal_()
        stock,context = torch.randn(7,1,158),torch.randn(7,76)
        logits = model(stock,context)["mixture_logits"]
        changed = context.clone()
        changed[:,:13]+=100.
        torch.testing.assert_close(logits,model(stock+12,changed)["mixture_logits"],rtol=0,atol=0)
        self.assertGreater(float((logits-model(stock,context+2.)["mixture_logits"]).abs().max().detach()),0.)
        source = {**config,"market":"csi300","seed":0}
        frozen = frozen_alpha_config({"market":"csi300","seed":2,"alpha_source":source})
        self.assertEqual(frozen["seed"],2)

    def test_invalid_parameters_targets_and_quantile_levels_fail(self):
        means,variances,logits = torch.zeros(3,2),torch.zeros(3,2),torch.zeros(3,2)
        for bad in [torch.zeros(3,1),torch.zeros(2)]:
            with self.assertRaises(ValueError):
                gaussian_mixture_nll(means,variances,logits,bad)
        with self.assertRaises(ValueError):
            gaussian_mixture_moments(means,variances[:,:1],logits)
        for levels in [[],[0.,0.5],[1.],[float("nan")]]:
            with self.assertRaises(ValueError):
                gaussian_mixture_quantiles(means,variances,logits,levels)
        for kwargs in [{"mixture_components":0},{"mixture_components":True},{"mixture_temperature":0.},
                       {"mixture_gate_input":"labels"}]:
            with self.assertRaises(ValueError):
                make_model({"family":"mixture_gaussian",**kwargs})

    def test_validation_diagnostics_use_true_mixture_quantiles_and_feature_only_forecasts(self):
        config = {"family":"mixture_gaussian","market":"sp500","mixture_components":2,
                  "width":32,"depth":1,"dropout":0.}
        model = make_model(config).eval()
        with torch.no_grad():
            model.output[-1].weight.zero_()
            model.output[-1].weight[1::2].normal_(std=0.02)
            model.output[-1].bias.copy_(torch.tensor([-4.,-3.,4.,-3.]))
        stock,context = torch.randn(6,1,158),torch.randn(6,76)
        index = pd.MultiIndex.from_product([pd.to_datetime(["2021-01-04","2021-01-05"]),["A","B","C"]],names=["datetime","instrument"])
        class FeatureOnlyData:
            day_ids = [0,1]
            def __init__(self,*args,**kwargs):
                if args[1] != "valid":
                    raise AssertionError("Diagnostics accessed another split")
                self.index = index
            def inputs(self,day):
                return stock[day*3:(day+1)*3],context[day*3:(day+1)*3]
            def selected_positions(self):
                return np.arange(6)
            def batch(self,*args):
                raise AssertionError("Prediction tried to load labels")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root/"config.json",config)
            write_json(root/"target_transform.json",{"scale":0.02,"kind":"raw_standardized"})
            torch.save({"model":model.state_dict(),"epoch":0},root/"best.pt")
            with patch("research.diagnostics.DailyData",FeatureOnlyData),patch("research.diagnostics.raw_labels",return_value=pd.Series([-0.08,0.,0.08,-0.07,0.,0.07],index=index)):
                first = diagnose(root,root/"first")
            with patch("research.diagnostics.DailyData",FeatureOnlyData),patch("research.diagnostics.raw_labels",return_value=pd.Series(np.arange(6)+10.,index=index)):
                diagnose(root,root/"changed_labels")
            frame = pd.read_pickle(root/"first/validation_forecasts.pkl")
            changed = pd.read_pickle(root/"changed_labels/validation_forecasts.pkl")
            for column in ["score","mean","sigma","q10","q50","q90","lower","upper"]:
                np.testing.assert_array_equal(frame[column],changed[column])
            np.testing.assert_array_equal(frame.lower,frame.q10)
            np.testing.assert_array_equal(frame.upper,frame.q90)
            self.assertGreater(float((frame.lower-(frame["mean"]-1.2815515655446004*frame.sigma)).abs().max()),0.01)
            np.testing.assert_allclose(first["mixture"]["mean_component_weights"],[0.5,0.5])
            self.assertEqual(sum(first["mixture"]["pit_histogram"]),6)


if __name__ == "__main__":
    unittest.main()
