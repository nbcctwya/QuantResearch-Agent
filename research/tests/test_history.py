"""Long histories must match native sampling and retain endpoint/label contracts."""
import copy
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from datasets.loader import DailyBatchSampler
from datasets.sampler import WindowSampler

from research.common import digest_file,write_json
from research.data import DailyData,window_row_indices
from research.history import prepare_history
from research.models import make_model,data_history_steps,loss_by_day,rank_loss


def sampler_fixture():
    dates = pd.date_range('2020-01-01',periods=40,freq='B')
    index = pd.MultiIndex.from_product([dates,['AAA','BBB']],names=['datetime','instrument'])
    # A long trading gap makes slicing a filled long window differ from native short-window fill.
    keep = ~((index.get_level_values('instrument')=='BBB') & index.get_level_values('datetime').isin(dates[8:24]))
    index = index[keep]
    columns = pd.MultiIndex.from_tuples([(name,str(i)) for name,width in [('feature',158),('prior',13),('market',63),('label',1)] for i in range(width)])
    values = np.random.default_rng(12).normal(size=(len(index),235)).astype('float32')
    return WindowSampler(pd.DataFrame(values,index=index,columns=columns),dates[0],dates[-1],8)


class HistoryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_native_long_windows_match_with_gaps_padding_and_poisoned_labels(self):
        sampler = sampler_fixture()
        positions = np.arange(len(sampler))
        for steps in [8,16,32]:
            sampler.step_len = steps
            actual = sampler.data_arr[window_row_indices(sampler,positions),:158]
            np.testing.assert_array_equal(actual,sampler[positions][:,:,:158])
            poisoned = copy.deepcopy(sampler)
            poisoned.data_arr[:,234] = 1e8
            np.testing.assert_array_equal(actual,poisoned[positions][:,:,:158])
        long = sampler[positions][:,-8:,:158]
        sampler.step_len = 8
        self.assertFalse(np.array_equal(long,sampler[positions][:,:,:158]))

    def test_cache_matches_all_endpoints_labels_and_native_history_and_reuses_without_fitting(self):
        sampler = sampler_fixture()
        order = DailyBatchSampler(sampler,shuffle=False).ordered_indices()
        index = sampler.get_index()[order]
        original = sampler[order]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'experiment/datasets/processed/csi300/train.pkl'
            source.parent.mkdir(parents=True)
            source.write_bytes(pickle.dumps(sampler))
            cache = root/'artifacts/cache/csi300/train'
            cache.mkdir(parents=True)
            np.save(cache/'stock.npy',original[:,:,:158])
            np.save(cache/'features.npy',original[:,-1,:234])
            np.save(cache/'label.npy',original[:,-1,234])
            index.to_frame(index=False).to_pickle(cache/'index.pkl')
            dates = index.get_level_values('datetime').to_numpy()
            boundaries = np.r_[0,np.flatnonzero(dates[1:]!=dates[:-1])+1,len(index)]
            np.save(cache/'boundaries.npy',boundaries)
            np.save(cache/'dates.npy',dates[boundaries[:-1]])
            write_json(cache/'manifest.json',{'source_sha256':digest_file(source)})
            before = {name:digest_file(cache/name) for name in ['stock.npy','features.npy','label.npy','index.pkl']}
            with patch('research.history.ARTIFACTS',root/'artifacts'),patch('research.history.EXPERIMENT_ROOT',root/'experiment'),patch('research.data.ARTIFACTS',root/'artifacts'):
                for steps in [16,32]:
                    path = prepare_history('csi300','train',steps)
                    sampler.step_len = steps
                    np.testing.assert_array_equal(np.load(path/'stock.npy'),sampler[order][:,:,:158])
                    data = DailyData('csi300','train',temporal=True,history_steps=steps,purge_days=5,limit_days=3)
                    pd.testing.assert_index_equal(data.index,index)
                    self.assertEqual(len(data.day_ids),3)
                    stock,context,labels = data.batch(0)
                    self.assertEqual(stock.shape,(2,steps,158))
                    np.testing.assert_array_equal(stock[:,-1].numpy(),original[:2,-1,:158])
                    np.testing.assert_array_equal(context.numpy(),original[:2,-1,158:234])
                    np.testing.assert_array_equal(labels.numpy(),original[:2,-1,234])
                    with patch('research.history.load_dataset',side_effect=AssertionError('Reloaded existing cache')):
                        self.assertEqual(prepare_history('csi300','train',steps),path)
                metadata = json.loads((cache/'history_32/manifest.json').read_text())
                self.assertEqual(metadata['history_steps'],32)
            self.assertEqual(before,{name:digest_file(cache/name) for name in before})

    def test_history_lengths_are_resolved_without_silently_changing_frozen_source_windows(self):
        temporal = {'family':'temporal_mixer','history_steps':16}
        risk = {'family':'mixture_gaussian','encoder':'temporal_mixer','history_steps':32}
        self.assertEqual(data_history_steps(temporal),16)
        self.assertEqual(data_history_steps({'family':'residual'}),8)
        self.assertEqual(data_history_steps({'family':'risk_overlay','alpha_source':{'family':'residual'},'risk_source':risk}),32)
        with self.assertRaisesRegex(ValueError,'separate native windows'):
            data_history_steps({'family':'risk_overlay','alpha_source':temporal,'risk_source':risk})
        for bad in [0,4,True,17]:
            with self.assertRaises(ValueError):
                make_model({'family':'temporal_mixer','history_steps':bad})
        with self.assertRaises(ValueError):
            make_model({'family':'master_control','history_steps':16})

    def test_explicit_eight_steps_preserve_initialization_predictions_and_gradients(self):
        config = {'family':'temporal_mixer','width':32,'depth':1,'dropout':0.,'market_gate':True}
        torch.manual_seed(55)
        implicit = make_model(config)
        torch.manual_seed(55)
        explicit = make_model({**config,'history_steps':8})
        for name,value in implicit.state_dict().items():
            torch.testing.assert_close(value,explicit.state_dict()[name],rtol=0,atol=0)
        stock,context = torch.randn(7,8,158),torch.randn(7,76)
        first,second = implicit(stock,context),explicit(stock,context)
        torch.testing.assert_close(first,second,rtol=0,atol=0)
        first.sum().backward();second.sum().backward()
        for one,two in zip(implicit.parameters(),explicit.parameters()):
            torch.testing.assert_close(one.grad,two.grad,rtol=0,atol=0)

    def test_long_history_models_train_and_packed_losses_keep_dates_separate(self):
        for steps in [16,32]:
            torch.manual_seed(83)
            config = {'family':'temporal_mixer','history_steps':steps,'width':32,'depth':1,'dropout':0.}
            model = make_model(config)
            reference = copy.deepcopy(model)
            stock,context,labels = torch.randn(11,steps,158),torch.randn(11,76),torch.randn(11)
            labels[4:]+=4.
            actual = loss_by_day(model(stock,context),labels,[4,7],'mixed')
            expected = torch.stack([rank_loss(reference(stock[:4],context[:4]),labels[:4],'mixed'),
                                    rank_loss(reference(stock[4:],context[4:]),labels[4:],'mixed')])
            torch.testing.assert_close(actual,expected,atol=2e-6,rtol=1e-5)
            actual.mean().backward();expected.mean().backward()
            for one,two in zip(model.parameters(),reference.parameters()):
                torch.testing.assert_close(one.grad,two.grad,atol=2e-6,rtol=1e-5)
            self.assertGreater(float(model.time_mix[0].weight.grad.abs().sum()),0.)
            with self.assertRaisesRegex(ValueError,'configured native window'):
                model(stock[:,:8],context)

    def test_extended_test_history_is_rejected_before_source_or_labels_are_read(self):
        with tempfile.TemporaryDirectory() as directory,patch('research.history.ARTIFACTS',Path(directory)):
            with self.assertRaisesRegex(RuntimeError,'selection lock'):
                prepare_history('csi300','test',16)


if __name__=='__main__':
    unittest.main()
