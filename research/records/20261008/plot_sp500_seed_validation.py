"""Render the accompanying validation curves; does not load data or model labels."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

matplotlib.rcParams["svg.hashsalt"] = "kbs-sp500-seed-validation"


def main():
    directory = Path(__file__).resolve().parent
    frame = pd.read_csv(directory/"sp500_seed_validation_curves.csv", parse_dates=["datetime"])
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                             gridspec_kw={"height_ratios":[2, 1]}, layout="constrained")
    names = ["sp500_raw_excess_seeds012", "sp500_independent_regime_seeds012", "sp500_factor_regime_seeds012"]
    labels = ["Unpenalized alpha", "Independent risk + regime", "Factor risk + regime"]
    colors = ["#3268a8", "#d46b2c", "#24876c"]
    for name, label, color in zip(names, labels, colors):
        part = frame.loc[frame.method==name].set_index("datetime")
        drawdown = part.nav/np.maximum(1., part.nav.cummax())-1.
        axes[0].plot(part.index, part.nav, label=label, color=color, linewidth=1.4)
        axes[1].plot(part.index, drawdown, color=color, linewidth=1.2)
    axes[0].axhline(1., color="gray", linewidth=0.6, linestyle="--")
    axes[0].set_ylabel("Portfolio value (initial = 1)")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[1].set_ylabel("Drawdown")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.))
    for ax in axes:
        ax.grid(alpha=0.2)
    axes[0].set_title("SP500: three-seed avg_none — VALIDATION ONLY (2021–2022)")
    axes[1].set_xlabel("Baseline Top30 / Ndrop5 / fees; mean scores are backtested anew")
    fig.savefig(directory/"sp500_seed_validation.png", dpi=170)
    svg = directory/"sp500_seed_validation.svg"
    fig.savefig(svg, metadata={"Date":None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines())+"\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
