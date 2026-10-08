"""Protect causal scoring, component identity, and the complete holdout comparison."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from research.common import config_id, digest_file, freeze_code, write_json
from research.ensembles import predict_blend, source_configs
from research.protocol import compare_baselines
from research.signals import blend_predictions
from research.train import require_locked_holdout


class SignalTests(unittest.TestCase):
    def test_launch_code_snapshot_remains_unchanged_after_workspace_edits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/"research").mkdir()
            (root/"research/__init__.py").write_text("")
            source = root/"research/model.py"
            source.write_text("value = 1\n")
            with patch("research.common.ROOT",root):
                first = freeze_code(root/"releases")
                self.assertEqual(freeze_code(root/"releases"),first)
                source.write_text("value = 2\n")
                second = freeze_code(root/"releases")
            self.assertNotEqual(first,second)
            self.assertEqual((first/"research/model.py").read_text(),"value = 1\n")
            self.assertEqual((second/"research/model.py").read_text(),"value = 2\n")

    def test_smoothing_is_causal_label_independent_and_resets_after_long_gaps(self):
        dates = pd.date_range("2020-01-01",periods=6,freq="B")
        index = pd.MultiIndex.from_tuples([(date,stock) for i,date in enumerate(dates)
                                          for stock in ["AAA","BBB"] if stock=="BBB" or i in [0,1,5]],
                                         names=["datetime","instrument"])
        frame = pd.DataFrame({"score":[0.,1.,4.,1.,1.,1.,1.,10.,1.],"label":0.},index=index)
        actual = blend_predictions([frame],[1.],normalization="none",alpha=0.5,max_gap=2)
        self.assertEqual(actual.loc[(dates[1],"AAA"),"score"],2.)
        self.assertEqual(actual.loc[(dates[5],"AAA"),"score"],10.)
        poisoned = frame.copy()
        poisoned["label"] = 987654321.
        label_changed = blend_predictions([poisoned],[1.],normalization="none",alpha=0.5,max_gap=2)
        np.testing.assert_equal(label_changed.score.to_numpy(),actual.score.to_numpy())
        poisoned.loc[(dates[5],"AAA"),"score"] = -1e9
        changed = blend_predictions([poisoned],[1.],normalization="none",alpha=0.5,max_gap=2)
        pd.testing.assert_series_equal(changed.loc[:dates[1],"score"],actual.loc[:dates[1],"score"])

    def test_score_mixture_rejects_coverage_or_label_mismatches(self):
        index = pd.MultiIndex.from_product([pd.date_range("2020-01-01",periods=2),["AAA","BBB"]],
                                           names=["datetime","instrument"])
        frame = pd.DataFrame({"score":[1.,2.,3.,4.],"label":[0.1,0.2,0.3,np.nan]},index=index)
        result = blend_predictions([frame,frame.copy()],[0.25,0.75],normalization="none")
        pd.testing.assert_frame_equal(result,frame)
        with self.assertRaises(ValueError):
            blend_predictions([frame,frame.iloc[:-1]],[0.5,0.5])
        changed = frame.copy()
        changed.iloc[0,changed.columns.get_loc("label")] = 1.
        with self.assertRaises(AssertionError):
            blend_predictions([frame,changed],[0.5,0.5])

    def test_components_follow_parent_seed_and_holdout_requires_exact_frozen_method(self):
        sources = [{"market":"csi300","family":"ridge","seed":0,"alpha":10.},
                   {"market":"csi300","family":"ridge","seed":0,"alpha":100.}]
        config = {"market":"csi300","family":"scores_blend","seed":1,"sources":sources,"weights":[0.5,0.5]}
        self.assertEqual([c["seed"] for c in source_configs(config)],[1,1])
        self.assertEqual([c["seed"] for c in sources],[0,0])
        with tempfile.TemporaryDirectory() as temporary,patch("research.train.ARTIFACTS",Path(temporary)):
            with self.assertRaises(RuntimeError):
                require_locked_holdout(config)
            write_json(Path(temporary)/"study/selection_lock.json",{
                "selected":{"csi300":[{**config,"seed":0}]},"seeds":[0,1]})
            require_locked_holdout(config)
            require_locked_holdout(source_configs(config)[0])
            with self.assertRaises(ValueError):
                require_locked_holdout({**config,"seed":2})
            with self.assertRaises(ValueError):
                require_locked_holdout({**config,"weights":[0.4,0.6]})

    def test_changed_component_checkpoint_is_rejected_before_test_inference(self):
        source = {"market":"csi300","family":"ridge","seed":0,"alpha":10.}
        config = {"market":"csi300","family":"scores_blend","seed":0,"sources":[source],"weights":[1.]}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trained = root/"trials/blend"
            checkpoint = root/"trials"/config_id(source)/"model.npz"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"first checkpoint")
            write_json(trained/"components.json",{
                "sources":[{"id":config_id(source),"config":source,"model_sha256":{"model.npz":digest_file(checkpoint)}}],
                "weights":[1.],"score_norm":"cs_z","ewm_alpha":1.,"max_gap":5})
            checkpoint.write_bytes(b"modified checkpoint")
            with patch("research.ensembles.ARTIFACTS",root),patch("research.train.predict_test") as inference:
                with self.assertRaises(ValueError):
                    predict_blend(config,trained,root/"fake_holdout")
                inference.assert_not_called()

    def test_missing_metrics_cannot_count_as_exceeding_all_baselines(self):
        keys = ["IC","ICIR","RankIC","RankICIR","AR","STD","MDD","Sharpe","Sortino","Calmar"]
        target = {key:1. for key in keys}
        with patch("research.protocol.baseline_envelope",return_value=target):
            comparison = compare_baselines({"AR":2.,"STD":0.5},"csi300")
        self.assertEqual(set(comparison),set(keys))
        self.assertTrue(comparison["AR"]["strictly_better"])
        self.assertFalse(comparison["RankIC"]["strictly_better"])
        self.assertFalse(all(row["strictly_better"] for row in comparison.values()))


if __name__ == "__main__":
    unittest.main()
