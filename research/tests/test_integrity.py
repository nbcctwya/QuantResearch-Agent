"""Check evaluation against published artifacts, and protect input/index integrity."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from datasets.sampler import WindowSampler

from research import ROOT
from research.data import DailyData, window_row_indices
from research.models import make_model,rank_loss,prediction_scores
from research.protocol import portfolio_metrics
from research.common import config_id,write_json


class IntegrityTests(unittest.TestCase):
    def test_metrics_reproduce_existing_baseline_curve(self):
        folder = ROOT/"references/baseline_results/FactorVAE-results"
        curve = pd.read_csv(folder/"curves/ensemble/csi300_factorvae.csv")
        expected = pd.read_csv(folder/"metrics/ensemble_metrics.csv")
        expected = expected.loc[expected.market == "csi300"].iloc[0]
        result = portfolio_metrics(curve.daily_ret_net)
        for name in ["AR","STD","MDD","Sharpe","Sortino","Calmar"]:
            self.assertAlmostEqual(result[name],expected[name],places=12)

    def test_window_gaps_match_qlib_and_labels_do_not_enter_inputs(self):
        dates = pd.date_range("2020-01-01",periods=12,freq="B")
        index = pd.MultiIndex.from_product([dates,["AAA","BBB"]],names=["datetime","instrument"])
        index = index.delete([0,2,5,9,12])
        groups = [(name,str(i)) for name,width in [("feature",158),("prior",13),("market",63),("label",1)] for i in range(width)]
        columns = pd.MultiIndex.from_tuples(groups)
        values = np.random.default_rng(0).normal(size=(len(index),235)).astype("float32")
        frame = pd.DataFrame(values,index=index,columns=columns)
        sampler = WindowSampler(frame.copy(),dates[0],dates[-1],8)
        positions = np.arange(len(sampler))
        actual = sampler.data_arr[window_row_indices(sampler,positions)]
        np.testing.assert_array_equal(actual,sampler[positions])
        changed = frame.copy()
        changed.loc[:,"label"] = 98765432.0
        poisoned = WindowSampler(changed,dates[0],dates[-1],8)
        np.testing.assert_array_equal(actual[:,:,:234],poisoned[positions][:,:,:234])

    def test_purge_and_date_stock_alignment(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)/"cache/csi300/train"
            directory.mkdir(parents=True)
            dates = pd.date_range("2020-01-01",periods=8,freq="B")
            index = pd.MultiIndex.from_product([dates,["AAA","BBB"]],names=["datetime","instrument"])
            index.to_frame(index=False).to_pickle(directory/"index.pkl")
            np.save(directory/"features.npy",np.arange(16*234,dtype="float32").reshape(16,234))
            np.save(directory/"label.npy",np.full(16,-999999,dtype="float32"))
            np.save(directory/"boundaries.npy",np.arange(0,17,2))
            np.save(directory/"dates.npy",dates.to_numpy())
            (directory/"manifest.json").write_text("{}")
            with patch("research.data.ARTIFACTS",Path(temporary)):
                data = DailyData("csi300","train",purge_days=5)
                self.assertEqual(len(data.day_ids),3)
                self.assertEqual(data.index[data.selected_positions()][-1],(dates[2],"BBB"))
                stock,context,labels = data.batch(0)
                self.assertEqual(stock.shape,(2,1,158))
                self.assertEqual(context.shape,(2,76))
                self.assertTrue((labels == -999999).all())
                self.assertFalse((stock == -999999).any())
                self.assertFalse((context == -999999).any())

    def test_cross_stock_aggregation_is_permutation_equivariant(self):
        torch.manual_seed(0)
        model = make_model({"family":"temporal_mixer","width":32,"depth":1,"dropout":0.0,
                            "latent_factors":4,"context":True,"market_gate":True}).eval()
        stock = torch.randn(9,8,158)
        context = torch.randn(1,76).expand(9,-1)
        order = torch.randperm(9)
        with torch.no_grad():
            expected = model(stock,context)[order]
            actual = model(stock[order],context[order])
        torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-5)

    def test_risk_forecast_trains_both_heads_and_inference_uses_only_inputs(self):
        torch.manual_seed(4)
        model = make_model({"family":"risk_aware","width":32,"depth":1,"dropout":0.0,
                            "context":True,"market_gate":True,"risk_penalty":0.25})
        stock,context = torch.randn(20,1,158),torch.randn(20,76)
        targets = torch.linspace(-2,2,20)
        predictions = model(stock,context)
        loss = rank_loss(predictions,targets,"gaussian_nll_rank")
        loss.backward()
        gradient = model.output[-1].weight.grad
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertTrue((gradient.abs().sum(1)>0).all())
        model.eval()
        with torch.no_grad():
            before = prediction_scores(model(stock,context))
            targets.fill_(1000000)
            after = prediction_scores(model(stock,context))
        torch.testing.assert_close(before,after)

    def test_quantiles_are_ordered_and_all_forecast_heads_receive_gradients(self):
        torch.manual_seed(12)
        model = make_model({"family":"quantile_aware","width":32,"depth":1,"dropout":0.0,
                            "context":True,"risk_penalty":0.25})
        predictions = model(torch.randn(20,1,158),torch.randn(20,76))
        quantiles = predictions["quantiles"]
        self.assertTrue((quantiles[:,0]<quantiles[:,1]).all())
        self.assertTrue((quantiles[:,1]<quantiles[:,2]).all())
        loss = rank_loss(predictions,torch.linspace(-2,2,20),"quantile_rank")
        loss.backward()
        gradient = model.output[-1].weight.grad
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertTrue((gradient.abs().sum(1)>0).all())

    def test_excess_target_removes_daily_market_component_using_training_rows(self):
        from research.train import standardized_return_targets
        index = pd.MultiIndex.from_product([pd.date_range("2020-01-01",periods=2),["AAA","BBB","CCC"]],
                                           names=["datetime","instrument"])
        returns = np.array([-0.03,0.01,0.02,-0.04,0.01,0.03])
        targets,scale = standardized_return_targets(returns,index,"raw_excess_standardized")
        shifted,new_scale = standardized_return_targets(returns+np.repeat([0.15,-0.2],3),index,
                                                        "raw_excess_standardized")
        np.testing.assert_allclose(shifted,targets,atol=1e-6)
        self.assertAlmostEqual(scale,new_scale,places=12)
        np.testing.assert_allclose(pd.Series(targets,index=index).groupby(level="datetime").mean(),0,atol=1e-6)

    def test_risk_overlay_freezes_variance_source_in_training(self):
        for family in ["risk_aware","factor_gaussian"]:
            source = {"market":"csi300","family":family,"width":32,"depth":1,
                      "seed":0,"dropout":0.5,"context":True}
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                path = root/"trials"/config_id(source)/"best.pt"
                path.parent.mkdir(parents=True)
                torch.save({"model":make_model(source).state_dict(),"config":source},path)
                with patch("research.models.ARTIFACTS",root):
                    model = make_model({"market":"csi300","family":"risk_overlay","width":32,"depth":1,
                                        "dropout":0.1,"context":True,"risk_source":source,"risk_penalty":0.1})
                model.train()
                self.assertTrue(model.alpha.training)
                self.assertFalse(model.risk_model.training)
                stock,context = torch.randn(20,1,158),torch.randn(20,76)
                loss = rank_loss(model(stock,context),torch.linspace(-2,2,20),"mixed")
                loss.backward()
                self.assertTrue(any(p.grad is not None for p in model.alpha.parameters()))
                self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.risk_model.parameters()))

    def test_interrupted_trial_is_queued_for_resume_and_completed_trial_is_not(self):
        from research.runner import recover_interrupted
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            unfinished = {"market":"csi300","family":"ridge","seed":0}
            completed = {"market":"sp500","family":"ridge","seed":0}
            first,second = config_id(unfinished),config_id(completed)
            write_json(root/"trials"/first/"config.json",unfinished)
            write_json(root/"trials"/second/"result.json",{"status":"complete"})
            state = {"phase":"validation_search","active":None,"pending":[first,second],"completed":[]}
            with patch("research.runner.ARTIFACTS",root),patch("research.runner.report"):
                recover_interrupted(state)
            self.assertEqual(state["pending"],[])
            self.assertEqual(state["resume_queue"],[unfinished])
            self.assertEqual(state["completed"],[second])


if __name__ == "__main__":
    unittest.main()
