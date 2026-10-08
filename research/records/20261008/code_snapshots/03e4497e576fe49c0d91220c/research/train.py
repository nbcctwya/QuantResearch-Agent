"""Run one fully recorded trial; selection uses validation data exclusively."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import random
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import ARTIFACTS
from .common import code_fingerprint, config_id, digest_file, locked_code_bundle, model_artifact_hashes, now, write_json
from .data import DailyData
from .averaging import ema_decay, make_ema
from .models import make_model, rank_loss, prediction_scores, uses_temporal_data, data_history_steps, can_pack_training_days, loss_by_day
from .numerical import fit_model_feature_encoders
from .risk_regime import (fit_frozen_risk_regime, frozen_alpha_config, record_risk_regime_diagnostics,
                          verify_frozen_sources)
from .opportunity import fit_frozen_alpha_opportunity, record_alpha_opportunity_diagnostics
from .protocol import evaluate_predictions, prediction_metrics, raw_labels


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(2)


def prediction_frame(data, predictions, label=None):
    positions = data.selected_positions()
    index = data.index[positions]
    if len(predictions) != len(index):
        raise ValueError("Prediction/index size mismatch")
    label = data.labels[positions] if label is None else label.reindex(index).to_numpy()
    return pd.DataFrame({"score": predictions, "label": label}, index=index)


def validation_score(frame):
    metrics = prediction_metrics(frame.score, frame.label)
    by_year = frame.groupby(frame.index.get_level_values("datetime").year)
    yearly = [prediction_metrics(part.score, part.label)["RankIC"] for _, part in by_year]
    # Penalize divergence across validation years; neither test scores nor baseline targets enter selection.
    score = metrics["RankIC"] - 0.15*float(np.std(yearly))
    return score, metrics


def standardized_return_targets(returns, index, kind):
    """Fit the target scale on the supplied training rows, optionally demean each date."""
    returns = np.asarray(returns,dtype=np.float64)
    if len(returns) != len(index) or not np.isfinite(returns).all():
        raise ValueError("Invalid selected raw training targets")
    if kind == "raw_excess_standardized":
        daily_mean = pd.Series(returns,index=index).groupby(level="datetime").transform("mean")
        returns = returns-daily_mean.to_numpy()
    elif kind != "raw_standardized":
        raise ValueError(f"Unknown raw return target: {kind}")
    scale = float(returns.std(ddof=1))
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("Degenerate training target scale")
    return np.clip(returns.astype(np.float32)/scale,-8,8).astype(np.float32),scale


def neural_predict(model, data, amp):
    model.eval()
    result = []
    with torch.no_grad():
        for day in data.day_ids:
            stock, context = data.inputs(day)
            with torch.autocast(device_type=data.device.type, dtype=torch.bfloat16,
                                enabled=amp and data.device.type == "cuda"):
                prediction = model(stock, context)
            result.append(prediction_scores(prediction).cpu().numpy())
    return np.concatenate(result)


def training_group_losses(model, data, days, config, weights, amp):
    batches = [data.batch(day) for day in days]
    objective = config.get("objective","mse")
    pack = len(days)>1 and can_pack_training_days(config)
    if pack:
        stock,context,labels = [torch.cat([batch[i] for batch in batches],dim=0) for i in range(3)]
        counts = [len(batch[2]) for batch in batches]
        with torch.autocast(data.device.type,dtype=torch.bfloat16,enabled=amp and data.device.type=="cuda"):
            predictions = model(stock,context)
        predictions = {key:value.float() for key,value in predictions.items()} if isinstance(predictions,dict) else predictions.float()
        return loss_by_day(predictions,labels.float(),counts,objective,weights[days])
    losses = []
    for day,(stock,context,labels) in zip(days,batches):
        with torch.autocast(data.device.type,dtype=torch.bfloat16,enabled=amp and data.device.type=="cuda"):
            predictions = model(stock,context)
        predictions = {key:value.float() for key,value in predictions.items()} if isinstance(predictions,dict) else predictions.float()
        losses.append(rank_loss(predictions,labels.float(),objective)*weights[day])
    return torch.stack(losses)


def train_neural(config, destination):
    decay = ema_decay(config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for neural trials; run the authorized worker with GPU access")
    temporal = uses_temporal_data(config)
    common = {"purge_days": config.get("purge_days", 5), "temporal": temporal, "device": "cuda",
              "history_steps":data_history_steps(config)}
    train = DailyData(config["market"], "train", train_start=config.get("train_start"),
                      limit_days=config.get("smoke_train_days"), **common)
    valid = DailyData(config["market"], "valid", limit_days=config.get("smoke_valid_days"), **common)
    target_kind = config.get("target_kind")
    if config["family"] in ("risk_aware","factor_gaussian","mixture_gaussian","quantile_aware") and target_kind is None:
        target_kind = "raw_standardized"
    if target_kind in ("raw_standardized","raw_excess_standardized"):
        selected = train.selected_positions()
        returns = raw_labels(config["market"],train.index[selected]).to_numpy(dtype=np.float32)
        targets,scale = standardized_return_targets(returns,train.index[selected],target_kind)
        train.labels = np.full(len(train.labels),np.nan,dtype=np.float32)
        train.labels[selected] = targets
        write_json(destination/"target_transform.json",{
            "kind":target_kind,"scale":scale,"fit_split":"selected purged training dates only",
            "demeaning":"selected training cross-section per date" if target_kind == "raw_excess_standardized" else None,
            "clipping":[-8,8],"train_samples":len(selected),"test_targets_used":False})
    free_bytes, _ = torch.cuda.mem_get_info()
    budget = int(free_bytes*0.7)
    used = train.preload(budget)
    valid.preload(max(0, budget-used))
    positions = valid.selected_positions()
    valid_labels = raw_labels(config["market"], valid.index[positions])
    last_path = destination / "last.pt"
    model = make_model(config)
    if not last_path.exists():
        fit_model_feature_encoders(model,train,config,destination)
    model = model.cuda()
    if temporal:
        write_json(destination/"history_model.json",{
            "history_steps":data_history_steps(config),"native_filling":"original Qlib sampler with changed local step_len",
            "input":"historical Alpha158; original latest JKP and market context",
            "endpoints":"original dates and stock membership; all latest Alpha158 rows checked exactly",
            "targets":"unchanged baseline five-day label; same purged training/validation date selection",
            "time_mixer":"linear time mixing and latest/four-step/full-window pooling; local adaptation, no full TimeMixer replication",
            "capacity":"temporal linear layer dimensions grow with history length",
            "prediction":"features only; test cache generated only after the existing holdout lock check"})
    if config["family"] == "mixture_gaussian":
        write_json(destination/"mixture_model.json",{
            "components":model.components,"temperature":model.temperature,"gate_input":model.gate_input,
            "architecture":"shared stock encoder; separate mean/variance output rows per Gaussian component",
            "gate_features":"63 market columns" if model.gate_input == "market" else "latest raw Alpha158 and 63 market columns",
            "initial_gate":"uniform weights; one component has no gate and matches independent Gaussian parameterization",
            "component_log_variance_clamp":[-6,4],"likelihood_dtype":"float64",
            "likelihood":"per-stock logsumexp of conditional Gaussian densities; normalizer omitted in training",
            "objective":"0.7 distribution NLL + 0.3 selected ranking loss; gaussian_nll uses only NLL",
            "variance":"weighted component variance plus weighted squared deviations of component means",
            "scoring":"mean / total_sigma ** risk_exponent - risk_penalty * total_sigma",
            "intervals":"true mixture CDF inversion for validation quantiles; no moment-matched normal interval",
            "interpretation":"local conditional mixture adaptation; no hard routing, VQ, or demonstrated semantic expert specialization",
            "prediction_inputs":"features only; future returns enter training loss and observed validation diagnostics only"})
    if config["family"] == "factor_gaussian":
        write_json(destination/"covariance_model.json",{
            "rank":config.get("factor_rank",4),"factor_loading_bound":config.get("factor_loading_bound",2.0),
            "covariance":"diag(exp(diagonal_log_variance)) + factor_loadings @ factor_loadings.T",
            "diagonal_log_variance_clamp":[-6,4],"likelihood_dtype":"float64",
            "training":"joint Gaussian NLL per date, normalized by stock count; labels enter the loss only",
            "factorization":"factor-sized Cholesky; posterior-residual quadratic; no dense stock covariance",
            "scoring":"mean / marginal_sigma ** risk_exponent - risk_penalty * marginal_sigma",
            "backtest":"unchanged baseline TopkDropoutStrategy; no covariance allocation replaces it"})
    if config["family"] == "risk_overlay":
        source_path = ARTIFACTS/"trials"/config_id(config["risk_source"])/"best.pt"
        write_json(destination/"risk_source.json",{"config":config["risk_source"],
                   "checkpoint_sha256":digest_file(source_path),"frozen":True,
                   "training":"only the alpha ranker is updated; the risk model retains its source seed"})
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("lr", 0.0005),
                                  weight_decay=config.get("weight_decay", 0.0001))
    epochs = config.get("epochs", 60)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs,
                                                          eta_min=config.get("lr",0.0005)*0.1)
    best_score, stale, start_epoch = -np.inf, 0, 0
    amp = config.get("amp", True)
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location="cuda", weights_only=False)
        if checkpoint["config"] != config:
            raise ValueError("Resume configuration mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        best_score, stale, start_epoch = checkpoint["best_score"], checkpoint["stale"], checkpoint["epoch"]+1
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])
        print(f"Resuming epoch {start_epoch+1}", flush=True)
    fit_frozen_risk_regime(model,train,config,destination,amp)
    averaged = make_ema(model,config)
    if averaged is not None:
        if last_path.exists():
            if "ema" not in checkpoint:
                raise ValueError("EMA resume checkpoint has no averaged weights or update counter")
            averaged.load_state_dict(checkpoint["ema"])
        write_json(destination/"weight_averaging.json",{
            "method":"EMA","decay":decay,"update":"after every training optimizer step; first update copies trained weights",
            "optimizer_model":"unaveraged model retains optimizer and scheduler state",
            "validation_and_best_checkpoint":"EMA weights, selected by unchanged validation ranking criterion",
            "buffers":"copy original training-fitted feature/regime buffers; no averaging or refitting",
            "resume":"raw training weights, EMA weights/update count, optimizer, scheduler and random states saved together",
            "inference":"ordinary model loaded from selected averaged parameters; features only",
            "interpretation":"parameter EMA adaptation; no Mean Teacher consistency loss or SWA learning-rate schedule"})
    half_life = config.get("half_life_years")
    days_per_update = config.get("days_per_update",1)
    if not isinstance(days_per_update,int) or days_per_update<1:
        raise ValueError("days_per_update must be a positive integer")
    write_json(destination/"batching.json",{
        "days_per_update":days_per_update,"packed_forward":days_per_update>1 and can_pack_training_days(config),
        "loss":"equal-weight mean of separate daily losses, preserving optional date age weights",
        "cross_stock_models":"separate forward passes for each date; one accumulated optimizer update",
        "optimizer_updates_per_epoch":int(np.ceil(len(train.day_ids)/days_per_update))})
    if config.get("objective") == "top30_pair":
        write_json(destination/"ranking_objective.json",{
            "name":"top30_pair","topk":30,
            "pair_target":"true selected training-date top 30 versus other stocks of the same date",
            "base_loss":"0.3 MSE + 0.3 (1 - Pearson correlation) + 0.4 mean weighted logistic pair loss",
            "pair_sampling":"all eligible pairs, deterministic",
            "ties":"fractional top-k membership at the boundary; no equal-return comparisons",
            "label_scope":"selected purged training rows only; labels never enter forecast inputs",
            "backtest":"unchanged baseline TopkDropoutStrategy, topk 30 and n_drop 5"})
    weights = np.ones(len(train.dates), dtype=float)
    if half_life:
        ages = (train.dates[train.day_ids].max()-train.dates)/np.timedelta64(1,"D")/365.25
        weights = np.exp(-np.log(2)*ages/half_life)
        weights /= weights[train.day_ids].mean()
    for epoch in range(start_epoch, epochs):
        begin = time.monotonic()
        model.train()
        losses = []
        order = np.random.permutation(train.day_ids)
        optimizer_steps = 0
        for start in range(0,len(order),days_per_update):
            days = order[start:start+days_per_update]
            optimizer.zero_grad(set_to_none=True)
            daily_losses = training_group_losses(model,train,days,config,weights,amp)
            loss = daily_losses.mean()
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if averaged is not None:
                averaged.update_parameters(model)
            optimizer_steps += 1
            losses.extend(daily_losses.detach().cpu().tolist())
        validation_model = averaged.module if averaged is not None else model
        predictions = neural_predict(validation_model, valid, amp)
        frame = prediction_frame(valid, predictions, valid_labels)
        score, metrics = validation_score(frame)
        if not np.isfinite(score):
            raise ValueError("Undefined validation selection metric")
        improved = score > best_score + 1e-5
        if improved:
            best_score, stale = score, 0
            torch.save({"model":validation_model.state_dict(), "config":config, "epoch":epoch,
                        "validation_score":score, "validation_metrics":metrics}, destination/"best.pt")
        else:
            stale += 1
        scheduler.step()
        progress = {"time":now(), "epoch":epoch+1, "train_loss":float(np.mean(losses)),
                    "valid_selection_score":score, "best_selection_score":best_score,
                    "valid":metrics, "seconds":time.monotonic()-begin, "stale":stale,
                    "optimizer_steps":optimizer_steps,"days_per_update":days_per_update}
        if averaged is not None:
            progress["ema_updates"] = int(averaged.n_averaged)
        with (destination/"epochs.jsonl").open("a") as stream:
            stream.write(json.dumps(progress)+"\n")
        write_json(destination/"progress.json", progress)
        print(json.dumps(progress), flush=True)
        checkpoint = {"config":config, "model":model.state_dict(), "optimizer":optimizer.state_dict(),
                      "scheduler":scheduler.state_dict(), "epoch":epoch, "best_score":best_score, "stale":stale,
                      "numpy_rng":np.random.get_state(), "torch_rng":torch.get_rng_state(),
                      "cuda_rng":torch.cuda.get_rng_state_all()}
        if averaged is not None:
            checkpoint["ema"] = averaged.state_dict()
        temporary = destination/"last.pt.tmp"
        torch.save(checkpoint, temporary)
        temporary.replace(last_path)
        if stale >= config.get("patience", 8):
            break
    checkpoint = torch.load(destination/"best.pt", map_location="cuda", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    frame = prediction_frame(valid, neural_predict(model, valid, amp), valid_labels)
    record_risk_regime_diagnostics(model,valid,destination,amp)
    return frame


def flat_arrays(data, config):
    positions = data.selected_positions()
    features = np.asarray(data.features[positions], dtype=np.float32)
    if not config.get("context", True):
        features = features[:, :158].copy()
    if config.get("cs_norm", False):
        offset = 0
        for day in data.day_ids:
            count = int(data.boundaries[day+1]-data.boundaries[day])
            stock = features[offset:offset+count,:158]
            stock -= stock.mean(0)
            stock /= np.maximum(stock.std(0),0.1)
            offset += count
    return features, np.asarray(data.labels[positions]), positions


def fit_frozen_overlay(config, destination):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for frozen neural scoring")
    alpha_source = frozen_alpha_config(config)
    source_path = ARTIFACTS/"trials"/config_id(alpha_source)
    if not (source_path/"result.json").exists():
        print(f"FIT_FROZEN_ALPHA {source_path.name} seed={alpha_source['seed']}",flush=True)
        run_training(alpha_source,source_path)
    result = json.loads((source_path/"result.json").read_text())
    if result["status"] != "complete" or result["config"] != alpha_source:
        raise ValueError("Frozen alpha result configuration mismatch")
    if result.get("smoke") and not (config.get("smoke_train_days") or config.get("smoke_valid_days")):
        raise ValueError("A full frozen scorer cannot use a smoke-trained alpha source")
    common = {"purge_days":config.get("purge_days",5),"temporal":uses_temporal_data(config),"device":"cuda",
              "history_steps":data_history_steps(config)}
    training = DailyData(config["market"],"train",train_start=config.get("train_start",alpha_source.get("train_start")),
                         limit_days=config.get("smoke_train_days"),**common)
    valid = DailyData(config["market"],"valid",limit_days=config.get("smoke_valid_days"),**common)
    budget = int(torch.cuda.mem_get_info()[0]*0.7)
    fit_statistics = config.get("risk_regime_strength",0.0) or config.get("alpha_opportunity_strength",0.0)
    used = training.preload(budget) if fit_statistics else 0
    valid.preload(max(0,budget-used))
    model = make_model(config).cuda().eval()
    checkpoint_path = destination/"best.pt"
    amp = config.get("amp",True)
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path,map_location="cuda",weights_only=False)
        if checkpoint["config"] != config:
            raise ValueError("Frozen scorer resume configuration mismatch")
        verify_frozen_sources(config,destination)
        model.load_state_dict(checkpoint["model"])
    else:
        fit_frozen_risk_regime(model,training,config,destination,amp)
        fit_frozen_alpha_opportunity(model,training,config,destination,amp)
        alpha_checkpoint = torch.load(source_path/"best.pt",map_location="cpu",weights_only=False)
        write_json(destination/"fixed_alpha.json",{
            "config":alpha_source,"id":config_id(alpha_source),"checkpoint_sha256":digest_file(source_path/"best.pt"),
            "epoch":alpha_checkpoint["epoch"]+1,"frozen":True,
            "scoring":"same alpha checkpoint reused; no alpha optimizer or new epoch selection",
            "seed_policy":"alpha source follows parent seed; variance source retains its configured seed"})
        risk_path = ARTIFACTS/"trials"/config_id(config["risk_source"])/"best.pt"
        write_json(destination/"risk_source.json",{"config":config["risk_source"],
                   "checkpoint_sha256":digest_file(risk_path),"frozen":True,
                   "training":"both alpha and risk networks are frozen; only training-feature regime statistics are fitted"})
        temporary = checkpoint_path.with_suffix(".pt.tmp")
        torch.save({"config":config,"model":model.state_dict(),"epoch":alpha_checkpoint["epoch"]},temporary)
        temporary.replace(checkpoint_path)
    labels = raw_labels(config["market"],valid.index[valid.selected_positions()])
    frame = prediction_frame(valid,neural_predict(model,valid,amp),labels)
    record_risk_regime_diagnostics(model,valid,destination,amp)
    record_alpha_opportunity_diagnostics(model,valid,destination,amp)
    return frame


def train_flat(config, destination):
    import lightgbm as lgb
    train = DailyData(config["market"], "train", purge_days=config.get("purge_days",5),
                      train_start=config.get("train_start"), limit_days=config.get("smoke_train_days"))
    valid = DailyData(config["market"], "valid", purge_days=config.get("purge_days",5),
                      limit_days=config.get("smoke_valid_days"))
    xtrain, ytrain, train_positions = flat_arrays(train, config)
    xvalid, yvalid, valid_positions = flat_arrays(valid, config)
    train_counts = np.diff(train.boundaries)[train.day_ids]
    valid_counts = np.diff(valid.boundaries)[valid.day_ids]
    weights = None
    if config.get("half_life_years"):
        age = (train.dates[train.day_ids].max()-train.dates[train.day_ids])/np.timedelta64(1,"D")/365.25
        weights = np.repeat(np.exp(-np.log(2)*age/config["half_life_years"]), train_counts)
    if config["family"] == "ridge":
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=config.get("alpha",1000.0), solver="cholesky")
        model.fit(xtrain, ytrain, sample_weight=weights)
        np.savez(destination/"model.npz", coef=model.coef_, intercept=model.intercept_)
        predictions = model.predict(xvalid)
    else:
        objective = config.get("objective","regression")
        is_ranker = objective in ("lambdarank", "rank_xendcg")
        if is_ranker:
            def relevance(labels, index):
                percentile = pd.Series(labels,index=index).groupby(level="datetime").rank(pct=True).to_numpy()
                return np.minimum((percentile*16).astype(np.int32),15)
            lgb_ytrain = relevance(ytrain,train.index[train_positions])
            lgb_yvalid = relevance(yvalid,valid.index[valid_positions])
        else:
            lgb_ytrain, lgb_yvalid = ytrain, yvalid
        params = {"objective":objective, "metric":"None", "verbosity":-1, "num_threads":4,
                  "learning_rate":config.get("lr",0.025), "num_leaves":config.get("num_leaves",31),
                  "max_depth":config.get("max_depth",-1), "min_data_in_leaf":config.get("min_data_in_leaf",500),
                  "feature_fraction":config.get("feature_fraction",0.8), "bagging_fraction":0.8, "bagging_freq":1,
                  "lambda_l1":config.get("lambda_l1",0.1), "lambda_l2":config.get("lambda_l2",10.0),
                  "max_bin":127, "seed":config["seed"], "deterministic":True, "force_col_wise":True}
        if is_ranker:
            params.update(label_gain=list(range(16)), lambdarank_truncation_level=40)
        training = lgb.Dataset(xtrain, label=lgb_ytrain, weight=weights,
                               group=train_counts if is_ranker else None)
        validation = lgb.Dataset(xvalid, label=lgb_yvalid, group=valid_counts if is_ranker else None, reference=training)
        starts = np.r_[0,np.cumsum(valid_counts)[:-1]]
        sums_y = np.add.reduceat(yvalid,starts)
        sums_y2 = np.add.reduceat(yvalid**2,starts)
        def rank_target_ic(prediction, _dataset):
            sums_p = np.add.reduceat(prediction, starts)
            covariance = np.add.reduceat(prediction*yvalid,starts)-sums_p*sums_y/valid_counts
            variance = (np.add.reduceat(prediction**2,starts)-sums_p**2/valid_counts)*(sums_y2-sums_y**2/valid_counts)
            return "daily_rank_target_ic",float(np.nanmean(covariance/np.sqrt(np.maximum(variance,1e-20)))),True
        model = lgb.train(params, training, num_boost_round=config.get("rounds",2000),
                          valid_sets=[validation], feval=rank_target_ic,
                          callbacks=[lgb.early_stopping(config.get("patience_rounds",150),first_metric_only=True),
                                     lgb.log_evaluation(100)])
        model.save_model(str(destination/"model.txt"))
        pd.DataFrame({"feature":np.arange(xtrain.shape[1]),"gain":model.feature_importance("gain")}).to_csv(destination/"importance.csv",index=False)
        predictions = model.predict(xvalid)
        write_json(destination/"fit.json", {"best_iteration":model.best_iteration,"params":params,
                                          "early_stopping":"mean daily Pearson against training-style cross-sectional rank targets"})
    labels = raw_labels(config["market"], valid.index[valid_positions])
    return prediction_frame(valid, predictions, labels)


def require_locked_holdout(config):
    path = ARTIFACTS/"study/selection_lock.json"
    if not path.exists():
        raise RuntimeError("New-model test inference requires a frozen selection lock")
    lock = json.loads(path.read_text())
    if config["seed"] not in lock.get("seeds",[0,1,2,3,4]):
        raise ValueError("Test seed is outside the frozen confirmation seeds")
    allowed = []
    for candidates in lock["selected"].values():
        for candidate in candidates:
            selected = {**candidate,"seed":config["seed"]}
            allowed.append(selected)
            if selected["family"] == "scores_blend":
                from .ensembles import source_configs
                allowed.extend(source_configs(selected))
    if config not in allowed:
        raise ValueError("Test configuration is not a frozen method or its component")
    if "code" in lock and code_fingerprint() != lock["code"]:
        raise ValueError("Test inference code differs from the frozen selection code")
    if "code_bundle" in lock and Path(__file__).resolve().parent != locked_code_bundle(lock)/"research":
        raise RuntimeError("Test inference must run from the verified frozen confirmation package")


def predict_test(config, trained, destination):
    require_locked_holdout(config)
    verify_frozen_sources(config,trained)
    if config["family"] == "scores_blend":
        from .ensembles import predict_blend
        metrics = predict_blend(config,trained,destination)
        write_json(destination/"test_run.json",{"config":config,"created_at":now(),
                   "trained_artifacts":model_artifact_hashes(trained),"code":code_fingerprint()})
        return metrics
    temporal = uses_temporal_data(config)
    device = "cuda" if config["family"] not in ("ridge","lgbm") else "cpu"
    test = DailyData(config["market"], "test", purge_days=0, temporal=temporal, device=device,
                     history_steps=data_history_steps(config))
    if config["family"] in ("ridge","lgbm"):
        x, _, _ = flat_arrays(test,config)
        if config["family"] == "ridge":
            params = np.load(trained/"model.npz")
            predictions = x@params["coef"]+params["intercept"]
        else:
            import lightgbm as lgb
            predictions = lgb.Booster(model_file=str(trained/"model.txt")).predict(x)
    else:
        checkpoint = torch.load(trained/"best.pt",map_location=device,weights_only=False)
        model = make_model(config).to(device)
        model.load_state_dict(checkpoint["model"])
        test.preload(int(torch.cuda.mem_get_info()[0]*0.6))
        predictions = neural_predict(model,test,config.get("amp",True))
    frame = prediction_frame(test,predictions)
    metrics = evaluate_predictions(frame,config["market"],destination,backtest=True)
    write_json(destination/"test_run.json",{"config":config,"created_at":now(),
               "trained_artifacts":model_artifact_hashes(trained),"code":code_fingerprint()})
    return metrics


def run_training(config,destination):
    destination.mkdir(parents=True,exist_ok=True)
    seed_all(config["seed"])
    started = time.monotonic()
    cpu_before = resource.getrusage(resource.RUSAGE_SELF)
    device = "coordinator" if config["family"] == "scores_blend" else ("cpu" if config["family"] in ("ridge","lgbm") else "cuda")
    write_json(destination/"config.json",config)
    write_json(destination/"run.json", {"status":"running","started_at":now(),"id":config_id(config),
                                  "code":code_fingerprint(),"device":device,
                                  "source_package":str(Path(__file__).resolve().parent),
                                  "versions":{name:importlib.metadata.version(name) for name in ["torch","pyqlib","numpy","pandas","scipy","scikit-learn","lightgbm"]},
                                  "selection":"validation only; 5 boundary days purged by default",
                                  "smoke":bool(config.get("smoke_train_days") or config.get("smoke_valid_days"))})
    if config["family"] == "scores_blend":
        from .ensembles import train_blend
        frame = train_blend(config,destination)
    elif config["family"] == "risk_overlay" and config.get("alpha_source"):
        frame = fit_frozen_overlay(config,destination)
    else:
        frame = train_flat(config,destination) if config["family"] in ("ridge","lgbm") else train_neural(config,destination)
    metrics = evaluate_predictions(frame,config["market"],destination/"valid",backtest=not config.get("skip_backtest",False))
    calibration = None
    if config["family"] in ("risk_aware","factor_gaussian","mixture_gaussian","quantile_aware"):
        from .diagnostics import diagnose
        diagnosis = diagnose(destination,destination/"valid/calibration")
        calibration = {key:diagnosis[key] for key in ["interval_80_coverage","uncertainty_abs_error_RankIC"]}
    score,_ = validation_score(frame)
    if not np.isfinite(score):
        raise ValueError("Undefined final validation selection metric")
    cpu_after = resource.getrusage(resource.RUSAGE_SELF)
    resources = {"cpu_seconds":cpu_after.ru_utime+cpu_after.ru_stime-cpu_before.ru_utime-cpu_before.ru_stime,
                 "peak_process_rss_mib":cpu_after.ru_maxrss/1024,
                 "gpu_peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_available() else None,
                 "gpu_peak_reserved_mib":torch.cuda.max_memory_reserved()/2**20 if torch.cuda.is_available() else None,
                 "peak_scope":"process lifetime; nested components share the process; parent time includes component fits"}
    write_json(destination/"result.json", {"status":"complete","completed_at":now(),"seconds":time.monotonic()-started,
                                     "config":config,"selection_score":score,"validation":metrics,
                                     "calibration":calibration,
                                     "resources":resources,
                                     "smoke":bool(config.get("smoke_train_days") or config.get("smoke_valid_days"))})
    run_metadata = json.loads((destination/"run.json").read_text())
    run_metadata.update(status="complete",completed_at=now())
    write_json(destination/"run.json",run_metadata)
    print(json.dumps({"complete":config_id(config),"seconds":time.monotonic()-started,"validation":metrics}),flush=True)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,type=Path)
    parser.add_argument("--out", required=True,type=Path)
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--trained", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.out.mkdir(parents=True,exist_ok=True)
    if args.test_only:
        if args.trained is None:
            raise ValueError("--trained is required for locked holdout evaluation")
        seed_all(config["seed"])
        metrics = predict_test(config,args.trained,args.out)
        print(json.dumps(metrics),flush=True)
        return
    run_training(config,args.out)


if __name__ == "__main__":
    main()
