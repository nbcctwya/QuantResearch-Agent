"""Recoverable validation-only seed confirmation before any holdout selection lock."""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import ARTIFACTS, ROOT
from .common import (code_fingerprint, config_id, digest_file, locked_code_bundle,
                     model_artifact_hashes, now, write_json)
from .protocol import FACTOR_CODE, METRIC_FILE, evaluate_predictions

METRICS = ["IC", "ICIR", "RankIC", "RankICIR", "AR", "STD", "MDD", "Sharpe", "Sortino", "Calmar"]
MARKETS = ["csi300", "sp500"]
CRITERION = "Pareto frontier first, then mean percentile rank across all ten actual validation-ensemble metrics; STD lower, all others higher"


def finite_metrics(metrics):
    return all(metrics.get(key) is not None and np.isfinite(metrics[key]) for key in METRICS)


def metric_ranks(records):
    if not records:
        return {}
    if not all(finite_metrics(row["validation"]) for row in records):
        raise ValueError("Selection requires all ten finite validation metrics")
    table = pd.DataFrame([{"id":row["id"], **{key:row["validation"][key] for key in METRICS}} for row in records]).set_index("id")
    ranks = pd.DataFrame({key:table[key].rank(pct=True, ascending=key != "STD") for key in METRICS})
    return {identifier:{"balanced":float(row.mean()), "minimum":float(row.min()), "metrics":row.to_dict()}
            for identifier, row in ranks.iterrows()}


def canonical_method(config):
    """Conservative no-op aliases; numerical score normalization stays distinct."""
    result = copy.deepcopy(config)
    if result.get("ema_decay") is None:
        result.pop("ema_decay", None)
    if result["family"] == "historical_risk" and result.get("risk_penalty", 0) == 0:
        from .risk_regime import frozen_alpha_config
        return canonical_method(frozen_alpha_config(result))
    if result["family"] == "risk_overlay":
        if result.get("risk_regime_strength", 0) == 0:
            result.pop("risk_regime_strength", None)
            result.pop("risk_regime_temperature", None)
        if result.get("alpha_opportunity_strength", 0) == 0:
            result.pop("alpha_opportunity_strength", None)
            result.pop("alpha_opportunity_temperature", None)
    for key in ["alpha_source", "risk_source"]:
        if key in result:
            result[key] = canonical_method(result[key])
    if result["family"] == "scores_blend":
        result["sources"] = [canonical_method(source) for source in result["sources"]]
    return result


def frame_fingerprint(frame):
    digest = hashlib.sha256()
    digest.update(json.dumps({"columns":list(frame.columns), "dtypes":[str(x) for x in frame.dtypes],
                              "index_names":frame.index.names}, sort_keys=True).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def check_frame(frame):
    if (frame.index.names != ["datetime", "instrument"] or not frame.index.is_unique
            or not frame.index.is_monotonic_increasing or list(frame.columns) != ["score", "label"]):
        raise ValueError("Validation confirmation requires sorted unique score/label predictions")
    if set(frame.index.get_level_values("datetime").year) != {2021, 2022}:
        raise ValueError("Validation confirmation cannot read another period")
    if not np.isfinite(frame.score.to_numpy()).all() or np.isinf(frame.label.to_numpy()).any():
        raise ValueError("Invalid validation confirmation predictions")


def shortlist(results, per_market=12):
    if isinstance(per_market, bool) or not isinstance(per_market, int) or per_market < 1:
        raise ValueError("Validation shortlist size must be a positive integer")
    selected, alias_report = {}, {}
    for market in MARKETS:
        rows = [row for row in results if row["config"]["market"] == market and row["config"]["seed"] == 0
                and not row.get("smoke") and finite_metrics(row["validation"])
                and row.get("selection_score") is not None and np.isfinite(row["selection_score"])]
        if not rows:
            raise ValueError("No eligible seed-0 validation candidates for "+market)
        balanced = metric_ranks(rows)
        table = pd.DataFrame([{"id":row["id"], "selection_score":row["selection_score"],
                               "AR":row["validation"]["AR"], "Sharpe":row["validation"]["Sharpe"]} for row in rows]).set_index("id")
        ranks = table.rank(pct=True)
        legacy = .5*ranks.selection_score+.25*ranks.AR+.25*ranks.Sharpe
        old_order = sorted(rows, key=lambda row:(-float(legacy.loc[row["id"]]), row["id"]))
        new_order = sorted(rows, key=lambda row:(-balanced[row["id"]]["balanced"], -balanced[row["id"]]["minimum"], row["id"]))
        # Alternate the existing search utility with the ten-metric utility.
        # Alias removal also verifies exact observed predictions and dtype.
        ordered = [item for pair in zip(old_order, new_order) for item in pair]
        seen, aliases, picked = {}, [], []
        for row in ordered:
            frame = pd.read_pickle(ARTIFACTS/"trials"/row["id"]/"valid/predictions.pkl")
            check_frame(frame)
            key = (config_id(canonical_method(row["config"])), frame_fingerprint(frame))
            if key in seen:
                if seen[key] != row["id"]:
                    aliases.append({"id":row["id"], "representative":seen[key], "exact_seed0_predictions_and_dtype":True})
                continue
            seen[key] = row["id"]
            picked.append(row)
            if len(picked) == per_market:
                break
        selected[market] = [{"id":row["id"], "config":row["config"], "seed0_validation":row["validation"],
                             "search_utility":float(legacy.loc[row["id"]]), "balanced_utility":balanced[row["id"]]}
                            for row in picked]
        alias_report[market] = list({(row["id"], row["representative"]):row for row in aliases}.values())
    return selected, alias_report


def protocol_inputs():
    paths = [METRIC_FILE, FACTOR_CODE/"generate_protocol_results.py"]
    paths += [ARTIFACTS/"cache"/market/split/"manifest.json" for market in MARKETS for split in ["train", "valid"]]
    if not all(path.exists() for path in paths):
        raise FileNotFoundError("Validation confirmation requires both baseline sources and four baseline cache manifests")
    return {"files":{str(path):digest_file(path) for path in paths},
            "versions":{name:importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "pyqlib", "torch"]}}


def verify_protocol_inputs(proof):
    if any(digest_file(name) != expected for name, expected in proof["files"].items()):
        raise ValueError("Frozen validation protocol or dataset manifest has changed")
    if any(importlib.metadata.version(name) != expected for name, expected in proof["versions"].items()):
        raise ValueError("Frozen validation numerical library version has changed")


def dependency_hashes(config, folder, visited=None):
    from .risk_regime import verify_frozen_sources
    visited = set() if visited is None else visited
    identifier = config_id(config)
    if identifier in visited:
        raise ValueError("Cyclic validation component graph")
    visited = visited | {identifier}
    verify_frozen_sources(config, folder)
    output = {}
    if config.get("risk_source"):
        record = json.loads((folder/"risk_source.json").read_text())
        if record["config"] != config["risk_source"]:
            raise ValueError("Frozen risk component configuration differs")
        checkpoint = ARTIFACTS/"trials"/config_id(record["config"])/"best.pt"
        if digest_file(checkpoint) != record["checkpoint_sha256"]:
            raise ValueError("Frozen risk checkpoint differs")
        output["risk_checkpoint"] = record["checkpoint_sha256"]
    if config["family"] == "scores_blend":
        from .ensembles import source_configs
        manifest = json.loads((folder/"components.json").read_text())
        if [row["config"] for row in manifest["sources"]] != source_configs(config):
            raise ValueError("Frozen blend component configurations differ")
        for source in manifest["sources"]:
            child = ARTIFACTS/"trials"/source["id"]
            if (source["id"] != config_id(source["config"]) or model_artifact_hashes(child) != source["model_sha256"]
                    or digest_file(child/"valid/predictions.pkl") != source["prediction_sha256"]):
                raise ValueError("Frozen blend component checkpoint or prediction differs")
            output[source["id"]] = dependency_hashes(source["config"], child, visited)
    return output


def artifact_record(config):
    folder = ARTIFACTS/"trials"/config_id(config)
    result = json.loads((folder/"result.json").read_text())
    if result["config"] != config or result["status"] != "complete" or result.get("smoke"):
        raise ValueError("Validation confirmation requires a complete matching full trial")
    if json.loads((folder/"config.json").read_text()) != config:
        raise ValueError("Validation trial config file differs")
    models = model_artifact_hashes(folder)
    if not models:
        raise FileNotFoundError("Validation trial has no reproducible model artifacts")
    run = json.loads((folder/"run.json").read_text())
    if run.get("status") != "complete":
        raise ValueError("Validation trial is still running")
    return {"id":config_id(config), "config":config, "model_sha256":models,
            "files":{name:digest_file(folder/name) for name in ["result.json", "run.json", "config.json", "valid/predictions.pkl"]},
            "dependencies":dependency_hashes(config, folder), "run":run}


def build_plan(results, settings, bundle, namespace="validation_confirmation"):
    if list((ARTIFACTS/"holdout").glob("*/test_run.json")):
        raise RuntimeError("Validation confirmation cannot be planned after new-model holdout observations")
    seeds = settings.get("validation_seeds", [0, 1, 2])
    final_seeds = settings.get("seeds", [0, 1, 2, 3, 4])
    if (len(seeds) != 3 or len(set(seeds)) != 3 or 0 not in seeds or len(final_seeds) != 5
            or len(set(final_seeds)) != 5 or not set(seeds) <= set(final_seeds)
            or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds+final_seeds)):
        raise ValueError("Confirmation requires three distinct validation seeds and five final seeds")
    if not namespace or namespace in {".", ".."} or Path(namespace).name != namespace:
        raise ValueError("Invalid validation confirmation namespace")
    count = settings.get("promotion_models_per_market", 3)
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("Promotion count must be a positive integer")
    candidates, aliases = shortlist(results, settings.get("validation_shortlist_per_market", 12))
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle/"manifest.json").read_text())
    if manifest["files"] != code_fingerprint():
        raise ValueError("Validation confirmation package differs from its executing code")
    existing = {}
    for market in MARKETS:
        for item in candidates[market]:
            for seed in seeds:
                config = {**item["config"], "seed":seed}
                if (ARTIFACTS/"trials"/config_id(config)/"result.json").exists():
                    existing[config_id(config)] = artifact_record(config)
    plan = {"created_at":now(), "scope":"purged 2021-2022 validation only; no selection lock or holdout query",
            "candidates":candidates, "aliases":aliases, "validation_seeds":seeds, "final_seeds":final_seeds,
            "promotion_models_per_market":settings.get("promotion_models_per_market", 3), "criterion":CRITERION,
            "namespace":namespace, "code_bundle":str(bundle), "code_manifest_sha256":digest_file(bundle/"manifest.json"),
            "code":manifest["files"], "protocol_inputs":protocol_inputs(), "preexisting":existing,
            "shortlist_criterion":"alternate seed-0 search utility and mean percentile of all ten metrics; conservative no-op aliases require exact predictions and dtype",
            "legacy_sources":"Preexisting fits retain their recorded original code, checkpoints and source package; the evaluation package is not attributed to earlier training."}
    locked_code_bundle(plan)
    return plan


def verify_plan(plan):
    locked_code_bundle(plan)
    if code_fingerprint() != plan["code"]:
        raise ValueError("Validation confirmation must use its frozen evaluation source")
    verify_protocol_inputs(plan["protocol_inputs"])
    if list((ARTIFACTS/"holdout").glob("*/test_run.json")):
        raise RuntimeError("Validation confirmation cannot run after observing new-model holdout")


def ensemble_validation(candidate, seeds, plan):
    verify_plan(plan)
    frames, sources = [], []
    for seed in seeds:
        config = {**candidate, "seed":seed}
        record = artifact_record(config)
        if record["id"] in plan["preexisting"] and record != plan["preexisting"][record["id"]]:
            raise ValueError("A preexisting validation fit has changed since the shortlist froze")
        if record["id"] not in plan["preexisting"] and (record["run"].get("code") != plan["code"]
                or record["run"].get("source_package") != str(Path(plan["code_bundle"])/"research")):
            raise ValueError("A new validation-confirmation fit did not use the frozen package")
        frame = pd.read_pickle(ARTIFACTS/"trials"/record["id"]/"valid/predictions.pkl")
        check_frame(frame)
        if frames:
            pd.testing.assert_index_equal(frame.index, frames[0].index)
            np.testing.assert_equal(frame.label.to_numpy(), frames[0].label.to_numpy())
        frames.append(frame)
        sources.append(record)
    combined = frames[0].copy()
    combined["score"] = np.mean([frame.score.to_numpy() for frame in frames], axis=0)
    name = config_id({**candidate, "seed":"validation_ensemble"})
    output = ARTIFACTS/"validation_checks"/plan["namespace"]/name
    provenance = {"candidate":candidate, "seeds":seeds, "method":"avg_none", "sources":sources,
                  "evaluation_code":plan["code"], "protocol_inputs":plan["protocol_inputs"]}
    cache = json.loads((output/"ensemble.json").read_text()) if (output/"ensemble.json").exists() else {}
    if cache.get("provenance") == provenance:
        for filename, expected in cache["files"].items():
            if digest_file(output/filename) != expected:
                raise ValueError("Validation ensemble cache has changed")
        pd.testing.assert_frame_equal(combined, pd.read_pickle(output/"predictions.pkl"), check_exact=True)
        metrics = json.loads((output/"metrics.json").read_text())
    else:
        metrics = evaluate_predictions(combined, candidate["market"], output, backtest=True)
        write_json(output/"ensemble.json", {"created_at":now(), "provenance":provenance,
                   "files":{filename:digest_file(output/filename) for filename in ["predictions.pkl", "metrics.json", "protocol.json", "curve.csv", "qlib_report.pkl", "daily_ranking.csv"]}})
    if not finite_metrics(metrics):
        raise ValueError("Validation ensemble has undefined selection metrics")
    return {"id":config_id(candidate), "config":candidate, "validation":metrics, "seeds":seeds,
            "method":"avg_none", "sources":sources, "rows":len(combined), "directory":str(output),
            "ensemble_record_sha256":digest_file(output/"ensemble.json")}


def confirmed_selection(records, per_market=3):
    if isinstance(per_market, bool) or not isinstance(per_market, int) or per_market < 1:
        raise ValueError("Promotion count must be a positive integer")
    selected, rankings = {}, {}
    for market in MARKETS:
        rows = [row for row in records if row["config"]["market"] == market]
        if not rows:
            raise ValueError("No complete validation ensemble for "+market)
        scores = metric_ranks(rows)
        evaluated = []
        for row in rows:
            dominated = any(other["id"] != row["id"] and all(
                other["validation"][key] <= row["validation"][key] if key == "STD" else other["validation"][key] >= row["validation"][key]
                for key in METRICS) and any(other["validation"][key] != row["validation"][key] for key in METRICS) for other in rows)
            evaluated.append({"id":row["id"], "config":row["config"], "validation":row["validation"],
                              "pareto_frontier":not dominated, **scores[row["id"]]})
        evaluated.sort(key=lambda row:(not row["pareto_frontier"], -row["balanced"], -row["minimum"], row["id"]))
        selected[market] = [row["config"] for row in evaluated[:per_market]]
        rankings[market] = evaluated
    return selected, rankings


def verify_confirmation_record(record):
    for source in record["sources"]:
        if artifact_record(source["config"]) != source:
            raise ValueError("A confirmed validation source has changed")
    output = Path(record["directory"])
    if digest_file(output/"ensemble.json") != record["ensemble_record_sha256"]:
        raise ValueError("A confirmed validation ensemble record has changed")
    proof = json.loads((output/"ensemble.json").read_text())
    for name, expected in proof["files"].items():
        if digest_file(output/name) != expected:
            raise ValueError("A confirmed validation ensemble artifact has changed")


def verify_selection_lock(lock, require_five_seeds=False):
    """Check validation evidence before confirmation or any new-model test read."""
    # Earlier lock formats remain readable for the existing compatibility checks.
    # All new coordinator locks include these validation-plan fields.
    if "validation_plan" not in lock:
        return
    verify_protocol_inputs(lock["protocol_inputs"])
    if digest_file(lock["validation_plan"]) != lock["validation_plan_sha256"]:
        raise ValueError("Frozen validation selection plan has changed")
    plan = json.loads(Path(lock["validation_plan"]).read_text())
    locked_code_bundle(plan)
    for key in ["code", "code_bundle", "code_manifest_sha256", "protocol_inputs"]:
        if plan[key] != lock[key]:
            raise ValueError("Selection lock differs from its validation plan")
    if lock["seeds"] != plan["final_seeds"] or lock["validation_seeds"] != plan["validation_seeds"]:
        raise ValueError("Selection lock seed sets differ from its validation plan")
    if digest_file(lock["validation_confirmation_report"]) != lock["validation_confirmation_sha256"]:
        raise ValueError("Frozen validation confirmation report has changed")
    report = json.loads(Path(lock["validation_confirmation_report"]).read_text())
    if not report["complete"] or report["plan"] != plan or report["selected"] != lock["selected"]:
        raise ValueError("Selection requires complete matching validation confirmation")
    selected, rankings = confirmed_selection(report["records"], plan["promotion_models_per_market"])
    if selected != lock["selected"] or rankings != report["rankings"]:
        raise ValueError("Selection is inconsistent with actual validation ensemble metrics")
    for record in report["records"]:
        if record["config"] in selected[record["config"]["market"]]:
            verify_confirmation_record(record)
    if require_five_seeds:
        proof = lock.get("five_seed_confirmation")
        if not proof:
            raise RuntimeError("Test inference requires complete five-seed confirmation evidence")
        if digest_file(proof["path"]) != proof["sha256"]:
            raise ValueError("Five-seed confirmation evidence has changed")
        confirmation = json.loads(Path(proof["path"]).read_text())
        if confirmation["selected"] != selected or confirmation["seeds"] != plan["final_seeds"]:
            raise ValueError("Five-seed confirmation differs from the frozen selection")
        expected = {config_id({**candidate, "seed":seed}) for candidates in selected.values()
                    for candidate in candidates for seed in plan["final_seeds"]}
        if {row["id"] for row in confirmation["records"]} != expected:
            raise ValueError("Five-seed confirmation evidence is incomplete")
        for row in confirmation["records"]:
            if artifact_record(row["config"]) != row:
                raise ValueError("A five-seed confirmation model or validation artifact has changed")


def finalize_five_seed_confirmation(lock, lock_path):
    """Attach checkpoint evidence once all selected five-seed fits are complete."""
    verify_selection_lock(lock)
    if "validation_plan" not in lock:
        return lock
    if lock.get("five_seed_confirmation"):
        verify_selection_lock(lock, require_five_seeds=True)
        return lock
    plan = json.loads(Path(lock["validation_plan"]).read_text())
    records = []
    for candidates in lock["selected"].values():
        for candidate in candidates:
            for seed in lock["seeds"]:
                record = artifact_record({**candidate, "seed":seed})
                if record["id"] in plan["preexisting"]:
                    if record != plan["preexisting"][record["id"]]:
                        raise ValueError("A preexisting selected fit has changed")
                elif (record["run"].get("code") != plan["code"] or
                      record["run"].get("source_package") != str(Path(plan["code_bundle"])/"research")):
                    raise ValueError("A new five-seed fit did not use the frozen package")
                records.append(record)
    path = ARTIFACTS/"study/five_seed_confirmation.json"
    write_json(path, {"created_at":now(), "selected":lock["selected"], "seeds":lock["seeds"],
                      "scope":"complete validation fits only; no holdout observation", "records":records})
    lock = {**lock, "five_seed_confirmation":{"path":str(path), "sha256":digest_file(path)}}
    write_json(lock_path, lock)
    verify_selection_lock(lock, require_five_seeds=True)
    return lock


def confirm_plan(plan, state, run_trial, report):
    """Resume candidate/seed fits and actual ensemble backtests without test access."""
    verify_plan(plan)
    records, failed = [], []
    progress_path = ARTIFACTS/"study"/(plan["namespace"]+"_progress.json")
    for market in MARKETS:
        for item in plan["candidates"][market]:
            missing = []
            for seed in plan["validation_seeds"]:
                config = {**item["config"], "seed":seed}
                identifier = config_id(config)
                if not (ARTIFACTS/"trials"/identifier/"result.json").exists():
                    if identifier in state["failed"]:
                        missing.append(identifier)
                        continue
                    if not run_trial(config, state, phase="validation_confirmation"):
                        if state.get("phase") == "stopped_by_control":
                            return False
                        if identifier not in state["failed"]:
                            state["failed"].append(identifier)
                        missing.append(identifier)
                    else:
                        artifact_record(config)
                if identifier not in missing and identifier not in state["completed"]:
                    state["completed"].append(identifier)
                report(state)
                if state.get("phase") == "stopped_by_control":
                    return False
            if missing:
                failed.append({"id":item["id"], "missing_or_failed":missing})
            else:
                records.append(ensemble_validation(item["config"], plan["validation_seeds"], plan))
            write_json(progress_path, {"updated_at":now(), "plan":plan, "scope":"validation only; test still sealed",
                       "complete":False, "records":records, "excluded":failed})
            report(state)
    if any(not any(row["config"]["market"] == market for row in records) for market in MARKETS):
        state["phase"] = "validation_confirmation_incomplete"
        report(state)
        return False
    verify_plan(plan)
    for record in records:
        verify_confirmation_record(record)
    selected, rankings = confirmed_selection(records, plan["promotion_models_per_market"])
    write_json(progress_path, {"updated_at":now(), "plan":plan, "scope":"validation only; test still sealed",
               "complete":True, "records":records, "excluded":failed, "selected":selected, "rankings":rankings})
    state["validation_confirmed"] = {"report":str(progress_path), "sha256":digest_file(progress_path), "selected":selected}
    report(state)
    return True
