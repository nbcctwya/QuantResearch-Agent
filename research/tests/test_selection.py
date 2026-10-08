"""Real signal averaging, direction-aware selection and restart/holdout gates."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import research
import research.runner as runner
import research.selection as selection
import research.train as train
from research.common import code_fingerprint, config_id, digest_file, write_json
from research.selection import (artifact_record, build_plan, canonical_method, check_frame,
                                confirm_plan, confirmed_selection, ensemble_validation,
                                finalize_five_seed_confirmation, frame_fingerprint,
                                metric_ranks, shortlist, verify_plan, verify_selection_lock)


def metrics(value=1.0):
    output = {key:float(value) for key in selection.METRICS}
    output.update(STD=.2, MDD=-.1)
    return output


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for module in [selection, runner, train]:
            context = patch.object(module, "ARTIFACTS", self.root)
            context.start()
            self.addCleanup(context.stop)
        for market in selection.MARKETS:
            for split in ["train", "valid"]:
                write_json(self.root/"cache"/market/split/"manifest.json", {"market":market, "split":split})
        self.bundle = self.root/"bundle"
        (self.bundle/"research").mkdir(parents=True)
        for name, expected in code_fingerprint().items():
            source = Path(research.__file__).parent/Path(name).name
            shutil.copy2(source, self.bundle/name)
            self.assertEqual(digest_file(self.bundle/name), expected)
        write_json(self.bundle/"manifest.json", {"files":code_fingerprint()})
        self.index = pd.MultiIndex.from_product([pd.to_datetime(["2021-01-04", "2022-01-04"]), ["A", "B", "C"]], names=["datetime", "instrument"])
        self.settings = {"validation_shortlist_per_market":1, "promotion_models_per_market":1,
                         "validation_seeds":[0, 1, 2], "seeds":[0, 1, 2, 3, 4]}
        self.rows = []
        for market in selection.MARKETS:
            for seed in [0, 1, 2]:
                config = {"market":market, "seed":seed, "family":"ridge", "alpha":1.0}
                row = self.write_trial(config)
                if seed == 0:
                    self.rows.append(row)

    def write_trial(self, config, dtype=np.float32, values=None):
        seed = config["seed"]
        values = [[3, 1, 2, 2, 1, 3], [0, 4, 1, 3, 1, 0], [1, 2, 5, 1, 3, 6]][seed] if values is None else values
        frame = pd.DataFrame({"score":np.array(values, dtype=dtype), "label":np.array([.1, -.1, .05]*2, dtype=np.float64)}, index=self.index)
        folder = self.root/"trials"/config_id(config)
        (folder/"valid").mkdir(parents=True, exist_ok=True)
        frame.to_pickle(folder/"valid/predictions.pkl")
        (folder/"model.npz").write_bytes(b"actual-model-"+str(seed).encode())
        result = {"id":config_id(config), "status":"complete", "config":config, "smoke":False,
                  "selection_score":1., "validation":metrics(10+seed)}
        write_json(folder/"config.json", config)
        write_json(folder/"result.json", result)
        write_json(folder/"run.json", {"status":"complete", "code":code_fingerprint(), "source_package":str(self.bundle/"research")})
        return result

    def plan(self):
        return build_plan(self.rows, self.settings, self.bundle)

    def evaluate(self, frame, market, output, backtest=True):
        self.assertTrue(backtest)
        output.mkdir(parents=True, exist_ok=True)
        frame.to_pickle(output/"predictions.pkl")
        write_json(output/"metrics.json", metrics(float(frame.score.iloc[0])))
        write_json(output/"protocol.json", {"scope":"unit fixture"})
        pd.DataFrame({"daily_ret_net":[.01, -.02]}).to_csv(output/"curve.csv", index=False)
        pd.DataFrame({"return":[.01, -.02]}).to_pickle(output/"qlib_report.pkl")
        pd.DataFrame({"RankIC":[.1, .2]}).to_csv(output/"daily_ranking.csv", index=False)
        return metrics(float(frame.score.iloc[0]))

    def lock(self):
        plan = self.plan()
        path = self.root/"study/validation_confirmation_plan.json"
        write_json(path, plan)
        state = {"completed":[], "failed":[], "phase":"validation_confirmation"}
        with patch.object(selection, "evaluate_predictions", side_effect=self.evaluate):
            self.assertTrue(confirm_plan(plan, state, lambda *args, **kwargs:self.fail("Unexpected refit"), lambda state:None))
        report = state["validation_confirmed"]
        lock = {"selected":report["selected"], "seeds":plan["final_seeds"],
                "validation_seeds":plan["validation_seeds"], "validation_plan":str(path),
                "validation_plan_sha256":digest_file(path),
                "validation_confirmation_report":report["report"], "validation_confirmation_sha256":report["sha256"],
                **{key:plan[key] for key in ["code", "code_bundle", "code_manifest_sha256", "protocol_inputs"]}}
        write_json(self.root/"study/selection_lock.json", lock)
        return lock

    def test_std_lower_and_negative_drawdown_higher_with_pareto_dominance(self):
        rows = []
        for market in selection.MARKETS:
            for name, std, drawdown in [("good", .1, -.02), ("bad", .3, -.2)]:
                value = metrics()
                value.update(STD=std, MDD=drawdown)
                rows.append({"id":market+name, "config":{"market":market, "family":name}, "validation":value})
        selected, ranked = confirmed_selection(rows, 1)
        for market in selection.MARKETS:
            self.assertEqual(selected[market][0]["family"], "good")
            self.assertTrue(ranked[market][0]["pareto_frontier"])
            self.assertFalse(ranked[market][1]["pareto_frontier"])

    def test_noop_aliases_preserve_normalization_and_active_regime(self):
        alpha = {"family":"residual", "market":"csi300", "seed":0}
        self.assertEqual(canonical_method({**alpha, "ema_decay":None}), alpha)
        historical = {"family":"historical_risk", "market":"csi300", "seed":0, "alpha_source":alpha, "risk_penalty":0.}
        self.assertEqual(canonical_method(historical), alpha)
        normalized = {"family":"scores_blend", "market":"csi300", "seed":0, "sources":[alpha], "score_norm":"cs_z"}
        self.assertNotEqual(canonical_method(normalized), alpha)
        risk = {"family":"risk_overlay", "market":"csi300", "seed":0, "risk_regime_strength":0., "risk_regime_temperature":2.}
        self.assertNotIn("risk_regime_temperature", canonical_method(risk))
        self.assertIn("risk_regime_temperature", canonical_method({**risk, "risk_regime_strength":.5}))

    def test_shortlist_aliases_require_exact_predictions_and_dtype(self):
        rows = self.rows.copy()
        for row in self.rows:
            alias = {**row["config"], "ema_decay":None}
            rows.append(self.write_trial(alias))
        selected, aliases = shortlist(rows, 2)
        self.assertTrue(all(len(selected[market]) == 1 for market in selection.MARKETS))
        self.assertTrue(all(aliases[market] for market in selection.MARKETS))
        for row in self.rows:
            self.write_trial({**row["config"], "ema_decay":None}, dtype=np.float64)
        selected, _ = shortlist(rows, 2)
        self.assertTrue(all(len(selected[market]) == 2 for market in selection.MARKETS))

    def test_plan_rejects_observed_holdout_invalid_seeds_and_counts(self):
        for changes in [{"validation_seeds":[0, 0, 1]}, {"validation_seeds":[True, 1, 2]},
                        {"seeds":[0, 1, 2]}, {"promotion_models_per_market":0}]:
            with self.assertRaises(ValueError):
                build_plan(self.rows, {**self.settings, **changes}, self.bundle)
        write_json(self.root/"holdout/fake/test_run.json", {"observed":True})
        with self.assertRaises(RuntimeError):
            self.plan()

    def test_actual_avg_none_and_cache_reuse_do_not_average_metrics(self):
        plan = self.plan()
        candidate = plan["candidates"]["csi300"][0]["config"]
        with patch.object(selection, "evaluate_predictions", side_effect=self.evaluate) as evaluate:
            first = ensemble_validation(candidate, [0, 1, 2], plan)
            second = ensemble_validation(candidate, [0, 1, 2], plan)
        self.assertEqual(evaluate.call_count, 1)
        self.assertEqual(first, second)
        actual = pd.read_pickle(Path(first["directory"])/"predictions.pkl")
        sources = [pd.read_pickle(self.root/"trials"/config_id({**candidate, "seed":seed})/"valid/predictions.pkl").score.to_numpy() for seed in [0, 1, 2]]
        np.testing.assert_array_equal(actual.score, np.mean(sources, axis=0))
        self.assertEqual(actual.score.dtype, np.dtype("float32"))
        self.assertNotEqual(first["validation"]["AR"], np.mean([10, 11, 12]))

    def test_checkpoint_or_protocol_manifest_tampering_stops_confirmation(self):
        plan = self.plan()
        candidate = plan["candidates"]["csi300"][0]["config"]
        (self.root/"trials"/config_id(candidate)/"model.npz").write_bytes(b"changed")
        with self.assertRaises(ValueError):
            ensemble_validation(candidate, [0, 1, 2], plan)
        write_json(self.root/"cache/sp500/train/manifest.json", {"changed":True})
        with self.assertRaises(ValueError):
            verify_plan(plan)

    def test_cached_ensemble_tampering_is_rejected(self):
        plan = self.plan()
        candidate = plan["candidates"]["csi300"][0]["config"]
        with patch.object(selection, "evaluate_predictions", side_effect=self.evaluate):
            record = ensemble_validation(candidate, [0, 1, 2], plan)
            (Path(record["directory"])/"curve.csv").write_text("changed")
            with self.assertRaises(ValueError):
                ensemble_validation(candidate, [0, 1, 2], plan)

    def test_mismatched_labels_or_coverage_and_nonvalidation_dates_fail(self):
        candidate = self.rows[0]["config"]
        folder = self.root/"trials"/config_id({**candidate, "seed":1})
        frame = pd.read_pickle(folder/"valid/predictions.pkl")
        frame.iloc[0, 1] += 1
        frame.to_pickle(folder/"valid/predictions.pkl")
        plan = self.plan()
        with self.assertRaises(AssertionError):
            ensemble_validation(candidate, [0, 1, 2], plan)
        frame.index = pd.MultiIndex.from_product([pd.to_datetime(["2023-01-04", "2024-01-04"]), ["A", "B", "C"]], names=frame.index.names)
        with self.assertRaises(ValueError):
            check_frame(frame)

    def test_partial_failure_excludes_candidate_and_never_creates_test_lock(self):
        for market in selection.MARKETS:
            folder = self.root/"trials"/config_id({"market":market, "seed":1, "family":"ridge", "alpha":1.0})
            shutil.rmtree(folder)
        plan = self.plan()
        state = {"completed":[], "failed":[], "phase":"validation_confirmation"}
        with patch.object(selection, "evaluate_predictions", side_effect=self.evaluate):
            self.assertFalse(confirm_plan(plan, state, lambda *args, **kwargs:False, lambda state:None))
        self.assertEqual(state["phase"], "validation_confirmation_incomplete")
        self.assertFalse((self.root/"study/selection_lock.json").exists())

    def test_complete_restart_reuses_results_and_ensemble_backtests(self):
        plan = self.plan()
        for _ in range(2):
            state = {"completed":[], "failed":[], "phase":"validation_confirmation"}
            with patch.object(selection, "evaluate_predictions", side_effect=self.evaluate) as evaluate:
                self.assertTrue(confirm_plan(plan, state, lambda *args, **kwargs:self.fail("Unexpected refit"), lambda state:None))
            self.assertEqual(evaluate.call_count, 2 if _ == 0 else 0)
            self.assertEqual(len(state["completed"]), 6)
        self.assertFalse((self.root/"study/selection_lock.json").exists())

    def test_new_fit_must_record_the_actual_frozen_package(self):
        candidate = self.rows[0]["config"]
        folder = self.root/"trials"/config_id({**candidate, "seed":1})
        shutil.rmtree(folder)
        plan = self.plan()
        self.write_trial({**candidate, "seed":1})
        write_json(folder/"run.json", {"status":"complete", "code":code_fingerprint(), "source_package":"wrong"})
        with self.assertRaises(ValueError):
            ensemble_validation(candidate, [0, 1, 2], plan)

    def test_lock_evidence_tampering_and_selection_changes_are_rejected(self):
        lock = self.lock()
        verify_selection_lock(lock)
        changed = copy.deepcopy(lock)
        changed["selected"]["csi300"][0]["alpha"] = 999.
        with self.assertRaises(ValueError):
            verify_selection_lock(changed)
        report = Path(lock["validation_confirmation_report"])
        report.write_text(report.read_text()+"\n")
        with self.assertRaisesRegex(ValueError, "confirmation report has changed"):
            verify_selection_lock(lock)

    def test_five_seed_proof_is_required_complete_idempotent_and_tamper_checked(self):
        lock = self.lock()
        with self.assertRaisesRegex(RuntimeError, "complete five-seed"):
            verify_selection_lock(lock, require_five_seeds=True)
        with self.assertRaises(FileNotFoundError):
            finalize_five_seed_confirmation(lock, self.root/"study/selection_lock.json")
        for row in self.rows:
            for seed in [3, 4]:
                self.write_trial({**row["config"], "seed":seed}, values=[1, 2, 3, 4, 5, 6])
        complete = finalize_five_seed_confirmation(lock, self.root/"study/selection_lock.json")
        verify_selection_lock(complete, require_five_seeds=True)
        path = Path(complete["five_seed_confirmation"]["path"])
        before = (digest_file(path), path.stat().st_mtime_ns)
        self.assertEqual(finalize_five_seed_confirmation(complete, self.root/"study/selection_lock.json"), complete)
        self.assertEqual((digest_file(path), path.stat().st_mtime_ns), before)
        candidate = {**self.rows[0]["config"], "seed":4}
        (self.root/"trials"/config_id(candidate)/"model.npz").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "five-seed confirmation model"):
            verify_selection_lock(complete, require_five_seeds=True)

    def test_new_five_seed_fit_requires_exact_frozen_training_package(self):
        lock = self.lock()
        for row in self.rows:
            for seed in [3, 4]:
                self.write_trial({**row["config"], "seed":seed}, values=[1, 2, 3, 4, 5, 6])
        config = {**self.rows[0]["config"], "seed":3}
        folder = self.root/"trials"/config_id(config)
        write_json(folder/"run.json", {"status":"complete", "code":code_fingerprint(), "source_package":"wrong"})
        with self.assertRaisesRegex(ValueError, "five-seed fit did not use"):
            finalize_five_seed_confirmation(lock, self.root/"study/selection_lock.json")

    def test_direct_test_entry_checks_confirmation_before_any_market_data(self):
        lock = self.lock()
        candidate = lock["selected"]["csi300"][0]
        with patch.object(train, "locked_code_bundle", return_value=Path(train.__file__).parent.parent), \
                patch.object(train, "DailyData") as data:
            with self.assertRaisesRegex(RuntimeError, "complete five-seed"):
                train.predict_test(candidate, self.root/"trials"/config_id(candidate), self.root/"holdout")
            data.assert_not_called()

    def test_failed_prelock_confirmation_cannot_create_selection_lock(self):
        plan = self.plan()
        write_json(self.root/"study/validation_confirmation_plan.json", plan)
        state = {"completed":[], "failed":[], "phase":"validation_confirmation"}
        with patch.object(runner, "control", return_value=self.settings), patch.object(runner, "report"), \
                patch.object(runner, "locked_code_bundle", return_value=Path(runner.__file__).parent.parent), \
                patch.object(selection, "confirm_plan", return_value=False):
            self.assertFalse(runner.promote(state))
        self.assertFalse((self.root/"study/selection_lock.json").exists())

    def test_coordinator_confirms_validation_then_five_seeds_before_ready(self):
        plan = self.plan()
        write_json(self.root/"study/validation_confirmation_plan.json", plan)
        for row in self.rows:
            for seed in [3, 4]:
                self.write_trial({**row["config"], "seed":seed}, values=[1, 2, 3, 4, 5, 6])
        state = {"completed":[], "failed":[], "phase":"validation_confirmation"}
        with patch.object(runner, "control", return_value=self.settings), patch.object(runner, "report"), \
                patch.object(runner, "locked_code_bundle", return_value=Path(runner.__file__).parent.parent), \
                patch.object(runner, "run_trial", side_effect=lambda *args, **kwargs:self.fail("Unexpected refit")), \
                patch.object(selection, "evaluate_predictions", side_effect=self.evaluate):
            self.assertTrue(runner.promote(state))
        lock = json.loads((self.root/"study/selection_lock.json").read_text())
        verify_selection_lock(lock, require_five_seeds=True)
        self.assertEqual(len(state["completed"]), 10)
        self.assertFalse(list((self.root/"holdout").glob("*/test_run.json")))

    def test_incomplete_five_seeds_do_not_report_ready_for_holdout(self):
        selected = {row["config"]["market"]:[row["config"]] for row in self.rows}
        write_json(self.root/"study/selection_lock.json", {"selected":selected, "seeds":[0, 1, 2, 3, 4]})
        state = {"completed":[], "failed":[], "phase":"five_seed_confirmation"}
        with patch.object(runner, "control", return_value={"stop":False}), patch.object(runner, "report"), \
                patch.object(runner, "locked_code_bundle", return_value=Path(runner.__file__).parent.parent), \
                patch.object(runner, "run_trial", return_value=False):
            self.assertFalse(runner.promote(state))
        self.assertEqual(state["phase"], "five_seed_confirmation_incomplete")

    def test_main_does_not_open_holdout_after_incomplete_confirmation(self):
        state = {"completed":[], "failed":[], "pending":[], "phase":"validation_confirmation_incomplete"}
        path = self.root/"study/state.json"
        write_json(path, state)
        with patch.object(runner, "STATE", path), patch.object(runner, "recover_interrupted"), \
                patch.object(runner, "report"), patch.object(runner, "control", return_value={"stop":False}), \
                patch.object(runner, "promote", return_value=False), patch.object(runner, "evaluate_holdout") as holdout, \
                patch("sys.argv", ["worker"]):
            runner.main()
        holdout.assert_not_called()


if __name__ == "__main__":
    unittest.main()
