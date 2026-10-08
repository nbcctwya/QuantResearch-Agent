"""Joint Gaussian loss is checked against dense distributions and independent daily graphs."""
import copy
import math
import unittest

import torch

from research.covariance import low_rank_gaussian_nll
from research.models import make_model,loss_by_day,rank_loss,can_pack_training_days


class CovarianceTests(unittest.TestCase):
    def test_joint_likelihood_and_all_gradients_match_dense_multivariate_normal(self):
        torch.manual_seed(54)
        mean=torch.randn(11,dtype=torch.float64,requires_grad=True)
        log_variance=torch.randn(11,dtype=torch.float64,requires_grad=True)
        loadings=(0.3*torch.randn(11,3,dtype=torch.float64)).requires_grad_()
        labels=torch.randn(11,dtype=torch.float64)
        actual=low_rank_gaussian_nll(mean,log_variance,loadings,labels)
        covariance=torch.diag(log_variance.exp())+loadings@loadings.T
        distribution=torch.distributions.MultivariateNormal(mean,covariance_matrix=covariance)
        expected=-distribution.log_prob(labels)/len(mean)-0.5*math.log(2*math.pi)
        torch.testing.assert_close(actual,expected,atol=1e-10,rtol=1e-10)
        actual_grad=torch.autograd.grad(actual,[mean,log_variance,loadings],retain_graph=True)
        expected_grad=torch.autograd.grad(expected,[mean,log_variance,loadings])
        for first,second in zip(actual_grad,expected_grad):
            torch.testing.assert_close(first,second,atol=1e-10,rtol=1e-10)

    def test_zero_factor_rank_is_the_independent_gaussian_likelihood(self):
        mean=torch.tensor([0.2,-0.1,0.7],requires_grad=True)
        log_variance=torch.tensor([-1.,0.3,-0.6],requires_grad=True)
        labels=torch.tensor([0.5,0.2,-0.1])
        actual=low_rank_gaussian_nll(mean,log_variance,torch.empty(3,0),labels)
        expected=torch.nn.functional.gaussian_nll_loss(mean,labels,log_variance.exp())
        torch.testing.assert_close(actual.float(),expected,atol=1e-6,rtol=1e-6)
        for first,second in zip(torch.autograd.grad(actual,[mean,log_variance],retain_graph=True),
                                torch.autograd.grad(expected,[mean,log_variance])):
            torch.testing.assert_close(first,second,atol=1e-6,rtol=1e-6)

    def test_large_common_shock_avoids_mahalanobis_cancellation(self):
        count,noise,loading,shock=64,1e-4,1e3,1e7
        actual=low_rank_gaussian_nll(torch.zeros(count,dtype=torch.float64),
                    torch.full((count,),math.log(noise),dtype=torch.float64),
                    torch.full((count,1),loading,dtype=torch.float64),
                    torch.full((count,),shock,dtype=torch.float64))
        eigenvalue=noise+count*loading**2
        expected=0.5*((count-1)*math.log(noise)+math.log(eigenvalue)+count*shock**2/eigenvalue)/count
        self.assertTrue(torch.isfinite(actual))
        self.assertAlmostEqual(float(actual),expected,places=7)

    def test_packed_predictions_preserve_per_date_covariance_and_gradients(self):
        torch.manual_seed(43)
        config={"family":"factor_gaussian","factor_rank":3,"width":32,"depth":1,"dropout":0.,
                "context":True,"market_gate":True,"risk_penalty":0.1}
        model=make_model(config)
        reference=copy.deepcopy(model)
        stock,context,labels=torch.randn(13,1,158),torch.randn(13,76),torch.randn(13)
        labels[5:]+=3.
        output=model(stock,context)
        actual=loss_by_day(output,labels,[5,8],"gaussian_nll_rank",[0.5,1.5])
        expected=torch.stack([rank_loss(reference(stock[:5],context[:5]),labels[:5],"gaussian_nll_rank")*0.5,
                              rank_loss(reference(stock[5:],context[5:]),labels[5:],"gaussian_nll_rank")*1.5])
        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
        actual.mean().backward()
        expected.mean().backward()
        for first,second in zip(model.parameters(),reference.parameters()):
            torch.testing.assert_close(first.grad,second.grad,atol=1e-5,rtol=1e-5)
        pooled=rank_loss(output,labels,"gaussian_nll_rank")
        self.assertGreater(abs(float(pooled.detach()-actual.mean().detach())),0.01)
        self.assertTrue(can_pack_training_days(config))
        self.assertFalse(can_pack_training_days({**config,"latent_factors":4}))

    def test_reported_marginal_variance_includes_factor_risk(self):
        torch.manual_seed(8)
        model=make_model({"family":"factor_gaussian","factor_rank":4,"width":32,"depth":1,"dropout":0.})
        output=model(torch.randn(9,1,158),torch.randn(9,76))
        expected=output["diagonal_log_variance"].exp()+output["factor_loadings"].square().sum(1)
        torch.testing.assert_close(output["log_variance"].exp(),expected)
        loss=rank_loss(output,torch.randn(9),"gaussian_nll_rank")
        loss.backward()
        head_gradient=model.output[-1].weight.grad
        for part in [head_gradient[:1],head_gradient[1:2],head_gradient[2:]]:
            self.assertTrue(torch.isfinite(part).all())
            self.assertGreater(float(part.abs().sum()),0.)


if __name__ == "__main__":
    unittest.main()
