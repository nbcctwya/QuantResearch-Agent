"""Direct reuse of baseline metrics and its Qlib TopkDropout backtest."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import ARTIFACTS, ROOT
from .common import digest_file, write_json

METRIC_FILE = ROOT / "references/baseline_code/AlphaMaster/src/alphamaster/evaluation_metrics.py"
FACTOR_CODE = ROOT / "references/baseline_code/FactorVAE"
spec = importlib.util.spec_from_file_location("pinned_baseline_metrics", METRIC_FILE)
baseline_metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline_metrics)
sys.path.insert(0, str(FACTOR_CODE))
from generate_protocol_results import run_standard_backtest  # noqa: E402
sys.path.remove(str(FACTOR_CODE))

prediction_metrics = baseline_metrics.prediction_metrics
portfolio_metrics = baseline_metrics.portfolio_metrics


def market_config(market):
    if market == "csi300":
        return {"provider_uri": str(Path.home()/".qlib/qlib_data/cn_data"), "region": "cn", "benchmark": "SH000300", "limit_threshold": 0.095}
    if market == "sp500":
        return {"provider_uri": str(Path.home()/".qlib/qlib_data/us_data"), "region": "us", "benchmark": "^gspc", "limit_threshold": None}
    raise ValueError(market)


def initialize_qlib(market):
    import qlib
    config = market_config(market)
    qlib.init(provider_uri=config["provider_uri"], region=config["region"], kernels=2,
              expression_cache=None, dataset_cache=None,
              exp_manager={"class":"MLflowExpManager", "module_path":"qlib.workflow.expm",
                           "kwargs":{"uri":f"file:{ARTIFACTS/'qlib_runs'}", "default_exp_name":"research"}})


def raw_labels(market, index):
    from qlib.data import D
    initialize_qlib(market)
    dates = index.get_level_values("datetime")
    path = ARTIFACTS / "raw_labels" / f"{market}_{dates.min():%Y%m%d}_{dates.max():%Y%m%d}.pkl"
    if path.exists():
        frame = pd.read_pickle(path)
    else:
        frame = D.features(D.instruments(market), ["Ref($close, -5) / Ref($close, -1) - 1"],
                           start_time=dates.min(), end_time=dates.max(), freq="day")
        frame = frame.iloc[:, 0].rename("label").swaplevel().sort_index()
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_pickle(path)
    return frame.reindex(index)


def evaluate_predictions(frame, market, destination, backtest=True):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if frame.index.names != ["datetime", "instrument"] or not frame.index.is_unique:
        raise ValueError("Predictions must have unique datetime/instrument keys")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("Predictions must be sorted")
    if not np.isfinite(frame["score"]).all():
        raise ValueError("Non-finite predictions")
    if np.isinf(frame["label"]).any():
        raise ValueError("Infinite labels")
    metrics = prediction_metrics(frame["score"], frame["label"])
    daily = frame.groupby(level="datetime", sort=True).apply(
        lambda x: pd.Series({"IC": x.score.corr(x.label), "RankIC": x.score.corr(x.label, method="spearman")}))
    daily.to_csv(destination / "daily_ranking.csv")
    for year in sorted(daily.index.year.unique()):
        metrics[f"RankIC_{year}"] = float(daily.loc[daily.index.year == year, "RankIC"].mean())
    frame.to_pickle(destination / "predictions.pkl")
    if backtest:
        from qlib.data import D
        initialize_qlib(market)
        dates = pd.DatetimeIndex(frame.index.get_level_values("datetime").unique())
        expected = pd.DatetimeIndex(D.calendar(start_time=dates.min(), end_time=dates.max()))
        if not dates.equals(expected):
            raise ValueError("Missing or extra trading dates in prediction coverage")
        report = run_standard_backtest(frame.score, market_config(market), str(dates.min().date()), str(dates.max().date()))
        if not pd.DatetimeIndex(report.index).equals(expected):
            raise ValueError("Qlib backtest calendar does not match requested period")
        report.to_pickle(destination / "qlib_report.pkl")
        curve = report[["return", "cost", "bench", "turnover"]].copy()
        curve["daily_ret_net"] = curve["return"] - curve["cost"]
        curve["nav"] = np.exp(np.log1p(curve.daily_ret_net).cumsum())
        curve.to_csv(destination / "curve.csv")
        metrics.update(portfolio_metrics(curve.daily_ret_net))
    import qlib
    write_json(destination / "metrics.json", metrics)
    write_json(destination / "protocol.json", {
        "metrics_source": str(METRIC_FILE.relative_to(ROOT)), "metrics_sha256": digest_file(METRIC_FILE),
        "backtest_source": str((FACTOR_CODE/"generate_protocol_results.py").relative_to(ROOT)),
        "backtest_source_sha256": digest_file(FACTOR_CODE/"generate_protocol_results.py"),
        "qlib_version": qlib.__version__, "market": market, "topk": 30, "n_drop": 5,
        "risk_degree": 0.95, "open_cost": 0.0005, "close_cost": 0.0015, "min_cost": 0,
        "net_return": "report.return - report.cost (deduct once)", "signal_shift": "Qlib internal t-1; no manual shift",
        "metric_convention": "baseline log1p return, ddof=1, 252 trading days, ICIR not annualized",
    })
    return metrics


def baseline_envelope(market):
    tables = [pd.read_csv(ROOT / "references/baseline_results/summary/metrics/ensemble_metrics.csv"),
              pd.read_csv(ROOT / "references/baseline_results/FSDM-results/metrics/ensemble_metrics.csv")]
    table = pd.concat(tables, ignore_index=True)
    table = table.loc[table.market == market]
    keys = ["IC", "ICIR", "RankIC", "RankICIR", "AR", "STD", "MDD", "Sharpe", "Sortino", "Calmar"]
    return {key: float(table[key].min() if key == "STD" else table[key].max()) for key in keys}


def compare_baselines(metrics, market):
    target = baseline_envelope(market)
    return {key: {"ours": metrics.get(key), "best_baseline": value,
                  "strictly_better": bool(metrics[key] < value if key == "STD" else metrics[key] > value)}
            for key, value in target.items() if key in metrics and metrics[key] is not None}
