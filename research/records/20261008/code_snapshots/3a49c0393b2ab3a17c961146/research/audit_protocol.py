"""Replay an existing baseline's predictions through the shared evaluator."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import ARTIFACTS, ROOT
from .common import write_json
from .protocol import evaluate_predictions
from generate_protocol_results import load_prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market",choices=["csi300","sp500"],required=True)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--prediction",type=Path,required=True)
    args = parser.parse_args()
    output = ARTIFACTS/"audit"/f"factorvae_{args.market}_seed{args.seed}"
    frame = load_prediction(args.prediction)
    actual = evaluate_predictions(frame,args.market,output)
    expected = pd.read_csv(ROOT/"references/baseline_results/FactorVAE-results/metrics/seed_metrics.csv")
    expected = expected.loc[(expected.market==args.market)&(expected.seed==args.seed)].iloc[0]
    keys = ["IC","ICIR","RankIC","RankICIR","AR","STD","MDD","Sharpe","Sortino","Calmar"]
    differences = {key:abs(actual[key]-float(expected[key])) for key in keys}
    write_json(output/"replay_comparison.json",{
        "market":args.market,"seed":args.seed,"differences":differences,
        "max_difference":max(differences.values()),"same_data_and_backtest":max(differences.values())<1e-10})
    np.testing.assert_allclose([actual[k] for k in keys],[expected[k] for k in keys],rtol=0,atol=1e-10)
    print(f"Reproduced {args.market} seed {args.seed}: max difference {max(differences.values()):.3g}",flush=True)


if __name__ == "__main__":
    main()
