"""Validation-only calibration diagnostics for return-distribution forecasts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import ARTIFACTS
from .common import code_fingerprint, digest_file, now, write_json
from .data import DailyData
from .models import make_model, uses_temporal_data
from .protocol import prediction_metrics, raw_labels


def diagnose(trained, destination):
    config = json.loads((trained/"config.json").read_text())
    if config["family"] not in ("risk_aware","factor_gaussian","quantile_aware"):
        raise ValueError("Calibration requires a conditional distribution model")
    transform = json.loads((trained/"target_transform.json").read_text())
    scale = transform["scale"]
    # The split is intentionally fixed; no test inference or test labels are accessed.
    data = DailyData(config["market"],"valid",purge_days=config.get("purge_days",5),
                     temporal=uses_temporal_data(config),device="cpu",
                     limit_days=config.get("smoke_valid_days"))
    torch.set_num_threads(2)
    model = make_model(config).eval()
    checkpoint = torch.load(trained/"best.pt",map_location="cpu",weights_only=False)
    model.load_state_dict(checkpoint["model"])
    pieces = []
    with torch.no_grad():
        for day in data.day_ids:
            stock,context = data.inputs(day)
            output = model(stock,context)
            columns = {"score":output["score"].numpy(),"mean":output["mean"].numpy()*scale}
            if "log_variance" in output:
                columns["sigma"] = np.exp(0.5*output["log_variance"].numpy())*scale
                if "factor_loadings" in output:
                    columns["idiosyncratic_sigma"] = np.exp(0.5*output["diagonal_log_variance"].numpy())*scale
                    columns["factor_sigma"] = output["factor_loadings"].square().sum(1).sqrt().numpy()*scale
            else:
                for i,name in enumerate(["q10","q50","q90"]):
                    columns[name] = output["quantiles"][:,i].numpy()*scale
            pieces.append(pd.DataFrame(columns))
    forecasts = pd.concat(pieces,ignore_index=True)
    forecasts.index = data.index[data.selected_positions()]
    forecasts["raw_return"] = raw_labels(config["market"],forecasts.index)
    forecasts["target"] = forecasts.raw_return
    if transform["kind"] == "raw_excess_standardized":
        # Realized validation means are ground truth for diagnostics, never prediction inputs.
        forecasts["target"] = forecasts.raw_return-forecasts.raw_return.groupby(level="datetime").transform("mean")
    forecasts["error"] = forecasts.target-forecasts["mean"]
    if not np.isfinite(forecasts.to_numpy()).all():
        raise ValueError("Non-finite validation forecasts or observations")
    if "sigma" in forecasts:
        forecasts["uncertainty"] = forecasts.sigma
        forecasts["lower"] = forecasts["mean"]-1.2815515655446004*forecasts.sigma
        forecasts["upper"] = forecasts["mean"]+1.2815515655446004*forecasts.sigma
    else:
        forecasts["uncertainty"] = forecasts.q90-forecasts.q10
        forecasts["lower"],forecasts["upper"] = forecasts.q10,forecasts.q90
    within_day = forecasts.uncertainty.groupby(level="datetime").rank(pct=True)
    forecasts["uncertainty_bin"] = np.minimum(np.floor(within_day.to_numpy()*5).astype(int),4)+1
    records = []
    for group,part in forecasts.groupby("uncertainty_bin"):
        records.append({"bin":int(group),"samples":len(part),
                        "predicted_uncertainty":float(part.uncertainty.mean()),
                        "observed_rmse":float(np.sqrt(np.mean(part.error**2))),
                        "observed_mae":float(part.error.abs().mean())})
    by_year = []
    for year,part in forecasts.groupby(forecasts.index.get_level_values("datetime").year):
        by_year.append({"year":int(year),"samples":len(part),
                        "rmse":float(np.sqrt(np.mean(part.error**2))),
                        "interval_80_coverage":float(((part.target>=part.lower)&(part.target<=part.upper)).mean())})
    result = {"created_at":now(),"market":config["market"],"family":config["family"],
              "split":"purged validation only","samples":len(forecasts),"days":len(data.day_ids),
              "target_kind":transform["kind"],"training_scale":scale,
              "scale_file_sha256":digest_file(trained/"target_transform.json"),
              "best_epoch":checkpoint["epoch"]+1,"inference":"CPU float32; diagnostic scores may differ slightly from CUDA bfloat16",
              "labels":"unclipped observed validation returns, demeaned by date for an excess-return target",
              "rmse":float(np.sqrt(np.mean(forecasts.error**2))),
              "mae":float(forecasts.error.abs().mean()),
              "interval_80_coverage":float(((forecasts.target>=forecasts.lower)&(forecasts.target<=forecasts.upper)).mean()),
              "uncertainty_abs_error_RankIC":prediction_metrics(forecasts.uncertainty,forecasts.error.abs())["RankIC"],
              "uncertainty_bins":records,"by_year":by_year,"code":code_fingerprint()}
    if "sigma" in forecasts:
        result["mean_normalized_squared_error"] = float(np.mean((forecasts.error/forecasts.sigma)**2))
        if "factor_sigma" in forecasts:
            result["mean_factor_variance_share"] = float((forecasts.factor_sigma**2/forecasts.sigma**2).mean())
            result["mean_idiosyncratic_sigma"] = float(forecasts.idiosyncratic_sigma.mean())
            result["mean_factor_sigma"] = float(forecasts.factor_sigma.mean())
    else:
        result["quantile_coverage"] = {name:float((forecasts.target<=forecasts[name]).mean())
                                        for name in ["q10","q50","q90"]}
    destination.mkdir(parents=True,exist_ok=True)
    forecasts.to_pickle(destination/"validation_forecasts.pkl")
    pd.DataFrame(records).to_csv(destination/"uncertainty_bins.csv",index=False)
    write_json(destination/"calibration.json",result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained",required=True,type=Path)
    parser.add_argument("--out",type=Path)
    args = parser.parse_args()
    destination = args.out or ARTIFACTS/"diagnostics"/args.trained.name
    result = diagnose(args.trained,destination)
    print(json.dumps({key:result[key] for key in ["market","family","samples","interval_80_coverage",
                                                "uncertainty_abs_error_RankIC","uncertainty_bins"]}),flush=True)


if __name__ == "__main__":
    main()
