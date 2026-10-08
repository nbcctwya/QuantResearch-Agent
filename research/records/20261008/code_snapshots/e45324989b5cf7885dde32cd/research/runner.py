"""Durable sequential study: validation search, five-seed confirmation, locked holdout."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import random
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import ARTIFACTS, ROOT
from .common import config_id, digest_file, freeze_code, locked_code_bundle, now, write_json

CONTROL = ROOT / "research/control.json"
STATE = ARTIFACTS / "study/state.json"


def control():
    return json.loads(CONTROL.read_text())


def initial_candidates():
    proposals = [
        {"family":"ridge", "alpha":1000.0, "context":False},
        {"family":"ridge", "alpha":10000.0, "context":True, "cs_norm":True},
        {"family":"residual", "width":128, "depth":2, "objective":"mse", "context":False},
        {"family":"residual", "width":128, "depth":2, "objective":"mixed", "context":True, "market_gate":True},
        {"family":"lgbm", "objective":"regression", "num_leaves":31, "min_data_in_leaf":1000, "context":False},
        {"family":"lgbm", "objective":"rank_xendcg", "num_leaves":31, "min_data_in_leaf":1000, "context":True},
        {"family":"batch_ensemble", "width":128, "depth":3, "members":8, "objective":"mixed", "context":False},
        {"family":"residual", "width":128, "depth":2, "objective":"tail_pair", "context":True, "market_gate":True, "cs_norm":True},
        {"family":"temporal_mixer", "width":128, "depth":2, "objective":"mixed", "context":True, "market_gate":True},
        {"family":"temporal_mixer", "width":128, "depth":2, "latent_factors":16, "objective":"mixed", "context":True, "market_gate":True},
        {"family":"batch_ensemble", "width":128, "depth":3, "members":8, "objective":"tail_pair", "context":True, "cs_norm":True},
        {"family":"master_control", "width":128, "dropout":0.3, "objective":"mse", "context":False},
        {"family":"lgbm", "objective":"regression", "num_leaves":15, "min_data_in_leaf":2000, "context":True, "half_life_years":3},
        {"family":"residual", "width":256, "depth":3, "objective":"corr", "context":True, "market_gate":True, "half_life_years":3},
        {"family":"temporal_mixer", "width":128, "depth":2, "latent_factors":8, "objective":"listwise", "context":True, "market_gate":True, "cs_norm":True},
    ]
    candidates = []
    for proposal in proposals:
        for market in ["csi300","sp500"]:
            config = {"market":market,"seed":0,"purge_days":5, **proposal}
            if config["family"] not in ("ridge","lgbm"):
                config.update(lr=0.0005, epochs=60, patience=8, dropout=config.get("dropout",0.1))
            candidates.append(config)
    return candidates


def read_results():
    results = []
    for file in (ARTIFACTS/"trials").glob("*/result.json"):
        row = json.loads(file.read_text())
        if row["status"] == "complete" and not row.get("smoke"):
            row["id"] = file.parent.name
            results.append(row)
    return results


def ranked_results(market):
    results = [r for r in read_results() if r["config"]["market"] == market and r["config"]["seed"] == 0
               and all(r.get("selection_score") is not None and r["validation"].get(key) is not None
                       for key in ["AR","Sharpe"])]
    if not results:
        return []
    frame = pd.DataFrame([{"id":r["id"],"selection_score":r["selection_score"],
                           "AR":r["validation"].get("AR",np.nan),
                           "Sharpe":r["validation"].get("Sharpe",np.nan)} for r in results]).set_index("id")
    ranks = frame.rank(pct=True)
    score = 0.5*ranks.selection_score+0.25*ranks.AR+0.25*ranks.Sharpe
    return sorted(results,key=lambda r:float(score.loc[r["id"]]),reverse=True)


def next_candidate(state):
    if state.get("resume_queue"):
        return state["resume_queue"].pop(0)
    seen = set(state["completed"]+state["failed"]+state["pending"])|{r["id"] for r in read_results()}
    extras_path = ROOT/"research/extra_candidates.json"
    extra = json.loads(extras_path.read_text()) if extras_path.exists() else []
    for proposal in extra+initial_candidates():
        if config_id(proposal) not in seen:
            return proposal
    # Adapt hyperparameters and objectives around validation leaders, retain new random proposals.
    rng = random.Random(state["proposals"]+9417)
    market = ["csi300","sp500"][state["proposals"]%2]
    leading = ranked_results(market)
    individuals = [r for r in leading if r["config"]["family"] != "scores_blend"
                   and not r["config"].get("alpha_source")][:12]
    frozen_scorers = [r for r in leading if r["config"].get("alpha_source")][:5]
    mixtures = [r for r in leading if r["config"]["family"] == "scores_blend"][:5]
    for _ in range(200):
        # Reserve most adaptive proposals for training methods; cheap mixtures must not consume the search.
        if len(individuals)>=2 and rng.random()<0.2:
            if frozen_scorers and rng.random()<0.5:
                config = dict(rng.choice(frozen_scorers)["config"])
            elif mixtures and rng.random()<0.5:
                config = dict(rng.choice(mixtures)["config"])
            else:
                members = rng.sample(individuals,2)
                config = {"market":market,"seed":0,"purge_days":5,"family":"scores_blend",
                          "objective":"signal_blend","sources":[r["config"] for r in members],"weights":[0.5,0.5]}
        elif individuals and rng.random() < 0.8:
            config = dict(rng.choice(individuals[:5])["config"])
        else:
            config = dict(rng.choice([p for p in initial_candidates() if p["market"]==market]))
        if config["family"] == "scores_blend":
            values = [rng.choice([0.25,0.5,0.75]) for _ in config["sources"]]
            config["weights"] = [v/sum(values) for v in values]
            config["score_norm"] = rng.choice(["cs_rank","cs_z","none"])
            config["ewm_alpha"] = rng.choice([0.25,0.5,0.75,1.0])
            config["max_gap"] = rng.choice([1,5])
        elif config["family"] == "historical_risk":
            config["risk_penalty"] = rng.choice([0.25,0.5,1.0])
            config["historical_risk_mode"] = rng.choice(["beta","total_volatility","idiosyncratic_volatility"])
        elif config.get("alpha_source"):
            config["risk_penalty"] = rng.choice([0.025,0.05,0.1,0.2])
            config["risk_regime_strength"] = rng.choice([0.0,0.5,1.0])
            config["risk_regime_temperature"] = rng.choice([0.5,1.0,2.0]) if config["risk_regime_strength"] else 1.0
            if config.get("alpha_opportunity_strength",0.0):
                config["alpha_opportunity_strength"] = rng.choice([0.25,0.5,1.0])
                config["alpha_opportunity_temperature"] = rng.choice([0.5,1.0,2.0])
        elif config["family"] in ("ridge","lgbm"):
            if config["family"] == "ridge":
                config["alpha"] = rng.choice([10.0,100.0,1000.0,10000.0,100000.0])
                config["context"] = rng.choice([True,False])
                config["cs_norm"] = rng.choice([True,False])
            else:
                config["num_leaves"] = rng.choice([15,31,63])
                config["min_data_in_leaf"] = rng.choice([300,1000,3000])
                config["objective"] = rng.choice(["regression","rank_xendcg","lambdarank"])
                config["lr"] = rng.choice([0.01,0.025,0.05])
        else:
            config["width"] = rng.choice([64,128,256])
            config["dropout"] = rng.choice([0.0,0.1,0.2,0.4])
            config["lr"] = rng.choice([0.0002,0.0005,0.001])
            config["objective"] = rng.choice(["mse","mixed","corr","tail_pair","top30_pair","listwise"])
            if config["family"] == "temporal_mixer":
                config["latent_factors"] = rng.choice([0,8,16,32])
                config["history_steps"] = rng.choice([8,16,32])
            if config["family"] == "mixture_gaussian":
                config["mixture_components"] = rng.choice([1,2,4,8])
                config["mixture_gate_input"] = rng.choice(["market","stock_market"])
                config["mixture_temperature"] = rng.choice([0.5,1.0,2.0])
                config["objective"] = rng.choice(["gaussian_nll_rank","gaussian_nll","corr","top30_pair"])
            if config["family"] == "student_t":
                config["student_df"] = rng.choice([3.0,5.0,10.0])
                config["objective"] = rng.choice(["student_nll_rank","student_nll","corr","top30_pair"])
        if config["family"] != "scores_blend" and not config.get("alpha_source"):
            config["half_life_years"] = rng.choice([None,2,3,5])
            config["train_start"] = rng.choice([None,"2013-01-01","2016-01-01"])
        if config_id(config) not in seen:
            return config
    raise RuntimeError("Candidate space exhausted")


def report(state):
    rows = []
    for result in read_results():
        config = result["config"]
        rows.append({"id":result["id"],"market":config["market"],"family":config["family"],
                     "seed":config["seed"],"objective":config.get("objective","mse"),
                     "selection_score":result["selection_score"],"seconds":result["seconds"],**result["validation"]})
    table = pd.DataFrame(rows)
    if len(table):
        table = table.sort_values(["market","selection_score"],ascending=[True,False])
    (ARTIFACTS/"study").mkdir(parents=True,exist_ok=True)
    table.to_csv(ARTIFACTS/"study/validation_leaderboard.csv",index=False)
    lines = ["# 自动实验状态", "",f"更新时间：{now()}",f"阶段：{state['phase']}",
             f"完整验证实验：{len(rows)}；worker 完成记录：{len(state['completed'])}；失败：{len(state['failed'])}。", "",
             "当前任务："+json.dumps(state.get("active"),ensure_ascii=False), "",
             "模型选择只使用验证集。以下数字均为验证集指标。", "",
             "|市场|模型|种子|RankIC|AR|Sharpe|", "|---|---|---|---|---|---|"]
    for row in rows:
        metrics = [row.get(key) for key in ["RankIC","AR","Sharpe"]]
        formatted = [f"{v:.5f}" if v is not None else "" for v in metrics]
        lines.append(f"|{row['market']}|{row['family']}|{row['seed']}|"+"|".join(formatted)+"|")
    (ARTIFACTS/"STATUS.md").write_text("\n".join(lines)+"\n")
    state["heartbeat"] = now()
    write_json(STATE,state)


def terminate_owned(process):
    if process.poll() is None:
        os.killpg(process.pid,signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL)
            process.wait()


def run_trial(config,state,phase="search",test_only=False):
    identifier = config_id(config)
    trained = ARTIFACTS/"trials"/identifier
    output = ARTIFACTS/"holdout"/identifier if test_only else trained
    output.mkdir(parents=True,exist_ok=True)
    print(f"START {phase} {identifier} {config['market']} {config['family']} seed={config['seed']}",flush=True)
    config_path = trained/"config.json"
    write_json(config_path,config)
    command = [sys.executable,"-u","-m","research.train","--config",str(config_path),"--out",str(output)]
    if test_only:
        command += ["--test-only","--trained",str(trained)]
    environment = dict(os.environ,OMP_NUM_THREADS="4",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="4",
                       PYTHONUNBUFFERED="1")
    if test_only or phase == "confirmation":
        frozen = json.loads((ARTIFACTS/"study/selection_lock.json").read_text())
        bundle = locked_code_bundle(frozen)
    elif phase == "validation_confirmation":
        plan_path = Path(state["validation_plan_path"])
        if digest_file(plan_path) != state["validation_plan_sha256"]:
            raise ValueError("Frozen validation confirmation plan has changed")
        bundle = locked_code_bundle(json.loads(plan_path.read_text()))
    else:
        bundle = freeze_code(ARTIFACTS/"code_releases")
    environment["KBS_RESEARCH_WORKSPACE_ROOT"] = str(ROOT)
    with (output/"console.log").open("a") as log:
        process = subprocess.Popen(command,cwd=bundle,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state["active"] = {"id":identifier,"market":config["market"],"family":config["family"],
                           "seed":config["seed"],"pid":process.pid,"phase":phase,"started_at":now(),"log":str(output/"console.log"),
                           "code_bundle":str(bundle)}
        report(state)
        started = time.monotonic()
        while process.poll() is None:
            settings = control()
            if settings.get("stop"):
                terminate_owned(process)
                state["phase"] = "stopped_by_control"
                report(state)
                return False
            if time.monotonic()-started > settings.get("max_trial_hours",4)*3600:
                terminate_owned(process)
                write_json(output/"failure.json",{"reason":"trial timeout","at":now()})
                break
            report(state)
            time.sleep(20)
        success = process.returncode == 0 and (output/("metrics.json" if test_only else "result.json")).exists()
        if not success:
            failure = json.loads((output/"failure.json").read_text()) if (output/"failure.json").exists() else {}
            failure.update(exit_code=process.returncode,at=now(),config=config)
            write_json(output/"failure.json",failure)
        state["active"] = None
        report(state)
    print(f"END {identifier} success={success}",flush=True)
    return success


def recover_interrupted(state):
    state.setdefault("failed",[])
    previous = state.get("active")
    if previous:
        command_path = Path(f"/proc/{previous['pid']}/cmdline")
        def owned_is_running():
            try:
                command = command_path.read_bytes()
                return b"research.train" in command and previous["id"].encode() in command
            except FileNotFoundError:
                return False
        started = datetime.fromisoformat(previous.get("started_at",now())).timestamp()
        # An orphaned trial can finish while the new worker keeps the journal alive.
        while owned_is_running():
            if time.time()-started > control().get("max_trial_hours",4)*3600:
                try:
                    if os.getpgid(previous["pid"]) == previous["pid"]:
                        os.killpg(previous["pid"],signal.SIGTERM)
                except ProcessLookupError:
                    pass
                deadline = time.monotonic()+15
                while owned_is_running() and time.monotonic()<deadline:
                    report(state)
                    time.sleep(1)
                if owned_is_running():
                    try:
                        if os.getpgid(previous["pid"]) == previous["pid"]:
                            os.killpg(previous["pid"],signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                while owned_is_running():
                    report(state)
                    time.sleep(1)
                folder = "holdout" if previous["phase"] == "holdout" else "trials"
                write_json(ARTIFACTS/folder/previous["id"]/"failure.json",{
                    "reason":"recovered trial timeout","at":now(),"active":previous})
                if previous["id"] not in state["failed"]:
                    state["failed"].append(previous["id"])
                break
            if control().get("stop"):
                if os.getpgid(previous["pid"]) == previous["pid"]:
                    os.killpg(previous["pid"],signal.SIGTERM)
                state["phase"]="stopped_by_control"
                report(state)
                return
            report(state)
            time.sleep(20)
        if previous["phase"] != "holdout" and previous["id"] not in state["pending"]:
            state["pending"].append(previous["id"])
    state["active"] = None
    state.setdefault("resume_queue",[])
    for identifier in list(state["pending"]):
        folder = ARTIFACTS/"trials"/identifier
        if (folder/"result.json").exists():
            if identifier not in state["completed"]:
                state["completed"].append(identifier)
        elif state["phase"] == "validation_search" and identifier not in state["failed"]:
            candidate = state.get("pending_configs",{}).get(identifier)
            if candidate is None and (folder/"config.json").exists():
                candidate = json.loads((folder/"config.json").read_text())
            if candidate is not None and candidate not in state["resume_queue"]:
                state["resume_queue"].append(candidate)
        state["pending"].remove(identifier)
        state.get("pending_configs",{}).pop(identifier,None)
    for row in read_results():
        if row["id"] not in state["completed"]:
            state["completed"].append(row["id"])
    report(state)


def promote(state):
    from .selection import (build_plan, confirm_plan, finalize_five_seed_confirmation,
                            verify_plan, verify_selection_lock)
    settings = control()
    lock_path = ARTIFACTS/"study/selection_lock.json"
    if not lock_path.exists():
        plan_path = ARTIFACTS/"study/validation_confirmation_plan.json"
        if not plan_path.exists():
            bundle = freeze_code(ARTIFACTS/"code_releases")
            plan = build_plan(read_results(), settings, bundle)
            write_json(plan_path, plan)
        plan = json.loads(plan_path.read_text())
        bundle = locked_code_bundle(plan)
        if state.get("validation_plan_sha256") and state["validation_plan_sha256"] != digest_file(plan_path):
            raise ValueError("Frozen validation confirmation plan has changed")
        state["validation_plan_path"] = str(plan_path)
        state["validation_plan_sha256"] = digest_file(plan_path)
        state["phase"] = "validation_confirmation"
        report(state)
        if Path(__file__).resolve().parent != bundle/"research":
            environment = dict(os.environ,KBS_RESEARCH_WORKSPACE_ROOT=str(ROOT),PYTHONUNBUFFERED="1")
            os.chdir(bundle)
            os.execvpe(sys.executable,[sys.executable,"-u","-m","research.runner"],environment)
        verify_plan(plan)
        if not confirm_plan(plan, state, run_trial, report):
            return False
        proof = state["validation_confirmed"]
        if digest_file(proof["report"]) != proof["sha256"]:
            raise ValueError("Validation confirmation report has changed")
        confirmation = json.loads(Path(proof["report"]).read_text())
        if not confirmation["complete"] or confirmation["selected"] != proof["selected"]:
            raise ValueError("Incomplete or inconsistent validation selection")
        verify_plan(plan)
        state["selected"] = proof["selected"]
        write_json(lock_path, {
            "locked_at":now(), "selected":state["selected"], "seeds":plan["final_seeds"],
            "criterion":plan["criterion"], "validation_seeds":plan["validation_seeds"],
            "validation_plan":str(plan_path), "validation_plan_sha256":digest_file(plan_path),
            "validation_confirmation_report":proof["report"], "validation_confirmation_sha256":proof["sha256"],
            "test_observed_before_lock":False, "code_bundle":plan["code_bundle"],
            "code_manifest_sha256":plan["code_manifest_sha256"], "code":plan["code"],
            "protocol_inputs":plan["protocol_inputs"]})
        report(state)
    frozen = json.loads(lock_path.read_text())
    verify_selection_lock(frozen)
    if state.get("selected") and state["selected"] != frozen["selected"]:
        raise ValueError("Journal selection differs from the authoritative selection lock")
    state["selected"] = frozen["selected"]
    bundle = locked_code_bundle(frozen)
    if Path(__file__).resolve().parent != bundle/"research":
        state["phase"] = "five_seed_confirmation"
        report(state)
        environment = dict(os.environ,KBS_RESEARCH_WORKSPACE_ROOT=str(ROOT),PYTHONUNBUFFERED="1")
        os.chdir(bundle)
        os.execvpe(sys.executable,[sys.executable,"-u","-m","research.runner"],environment)
    seeds = frozen.get("seeds", settings.get("seeds",[0,1,2,3,4]))
    state["phase"] = "five_seed_confirmation"
    all_complete = True
    for market, selected in state["selected"].items():
        for candidate in selected:
            for seed in seeds:
                config = {**candidate,"seed":seed}
                identifier = config_id(config)
                result_path = ARTIFACTS/"trials"/identifier/"result.json"
                if result_path.exists():
                    from .selection import artifact_record
                    artifact_record(config)
                    if identifier not in state["completed"]:
                        state["completed"].append(identifier)
                    continue
                if identifier in state["failed"]:
                    all_complete = False
                    continue
                if run_trial(config,state,phase="confirmation"):
                    state["completed"].append(identifier)
                else:
                    state["failed"].append(identifier)
                    all_complete = False
                report(state)
                if control().get("stop"):
                    return False
    if not all_complete:
        state["phase"] = "five_seed_confirmation_incomplete"
        report(state)
    else:
        finalize_five_seed_confirmation(frozen, lock_path)
    return all_complete


def evaluate_holdout(state):
    from .protocol import compare_baselines,evaluate_predictions
    from .selection import verify_selection_lock
    verify_selection_lock(json.loads((ARTIFACTS/"study/selection_lock.json").read_text()), require_five_seeds=True)
    state["phase"] = "locked_holdout_evaluation"
    settings = control()
    frozen = json.loads((ARTIFACTS/"study/selection_lock.json").read_text())
    seeds = frozen.get("seeds",settings.get("seeds",[0,1,2,3,4]))
    summary = []
    for market,selected in state["selected"].items():
        for candidate in selected:
            frames, records = [], []
            for seed in seeds:
                config = {**candidate,"seed":seed}
                identifier = config_id(config)
                if identifier not in state["completed"]:
                    continue
                target = ARTIFACTS/"holdout"/identifier
                if not (target/"metrics.json").exists() and not run_trial(config,state,phase="holdout",test_only=True):
                    continue
                metrics = json.loads((target/"metrics.json").read_text())
                records.append({"seed":seed,**metrics})
                frames.append(pd.read_pickle(target/"predictions.pkl"))
                report(state)
                if control().get("stop"):
                    return
            if len(frames) != len(seeds):
                continue
            reference = frames[0]
            for frame in frames[1:]:
                if not frame.index.equals(reference.index):
                    raise ValueError("Ensemble seed prediction coverage differs")
                np.testing.assert_equal(frame.label.to_numpy(),reference.label.to_numpy())
            ensemble = reference.copy()
            ensemble["score"] = np.mean([frame.score.to_numpy() for frame in frames],axis=0)
            identifier = config_id({**candidate,"seed":"ensemble"})
            destination = ARTIFACTS/"holdout"/identifier
            metrics = evaluate_predictions(ensemble,market,destination,backtest=True)
            write_json(destination/"ensemble.json",{"method":"avg_none","candidate":candidate,"seeds":seeds,
                       "selection_lock_sha256":digest_file(ARTIFACTS/"study/selection_lock.json"),
                       "code_manifest_sha256":frozen["code_manifest_sha256"],
                       "prediction_sha256":{str(seed):digest_file(ARTIFACTS/"holdout"/config_id({**candidate,"seed":seed})/"predictions.pkl")
                                            for seed in seeds}})
            comparison = compare_baselines(metrics,market)
            write_json(destination/"baseline_comparison.json",comparison)
            seeds_table = pd.DataFrame(records)
            seeds_table.to_csv(destination/"seed_metrics.csv",index=False)
            seeds_table.drop(columns="seed").agg(["mean","std"]).to_csv(destination/"seed_mean_std.csv")
            summary.append({"market":market,"id":identifier,"family":candidate["family"],
                            "seeds":seeds,"metrics":metrics,"comparison":comparison,
                            "strictly_exceeds_every_metric":len(comparison)==10 and all(v["strictly_better"] for v in comparison.values())})
            write_json(ARTIFACTS/"study/holdout_summary.json",summary)
            report(state)
    state["phase"] = "campaign_finished"
    state["goal_satisfied"] = all(any(r["market"]==market and r["strictly_exceeds_every_metric"] for r in summary)
                                 for market in ["csi300","sp500"])
    report(state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-trials",type=int,help="Development limit; stop after this many new full trials")
    args = parser.parse_args()
    ARTIFACTS.mkdir(parents=True,exist_ok=True)
    lock = (ARTIFACTS/"runner.lock").open("w")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if STATE.exists():
        state = json.loads(STATE.read_text())
    else:
        state = {"started_epoch":time.time(),"started_at":now(),"phase":"validation_search",
                 "completed":[],"failed":[],"pending":[],"proposals":0,"active":None}
    recover_interrupted(state)
    report(state)
    new_trials = 0
    while state["phase"] == "validation_search":
        settings = control()
        if settings.get("stop"):
            state["phase"]="stopped_by_control"
            report(state)
            return
        elapsed = time.time()-state["started_epoch"]
        search_hours = max(0,settings.get("hours",72)-settings.get("reserve_confirmation_hours",12))
        if elapsed >= search_hours*3600 or state["proposals"] >= settings.get("max_search_trials",180):
            break
        config = next_candidate(state)
        identifier = config_id(config)
        state["proposals"] += 1
        state["pending"].append(identifier)
        state.setdefault("pending_configs",{})[identifier]=config
        report(state)
        success = run_trial(config,state)
        state["pending"].remove(identifier)
        state["pending_configs"].pop(identifier,None)
        (state["completed"] if success else state["failed"]).append(identifier)
        report(state)
        new_trials += 1
        if args.max_trials is not None and new_trials >= args.max_trials:
            return
    if state["phase"] == "stopped_by_control":
        return
    if state["phase"] != "campaign_finished":
        ready = promote(state)
        if ready and not control().get("stop"):
            evaluate_holdout(state)


if __name__ == "__main__":
    main()
