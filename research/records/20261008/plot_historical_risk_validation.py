"""Standalone figures from completed baseline validation backtests."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from research import ROOT


def draw(table, suffix, title, normalization_controls=None):
    folder = ROOT/"research/records/20261008"
    colors = {"beta":"#2176ae", "total_volatility":"#c45a21", "idiosyncratic_volatility":"#44803f"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for row, market in enumerate(["csi300", "sp500"]):
        data = table.loc[table.market == market]
        for column, key in enumerate(["STD", "MDD"]):
            axis = axes[row, column]
            for mode, color in colors.items():
                points = data.loc[(data.risk_mode == mode) & (data.risk_penalty > 0)].sort_values("risk_penalty")
                axis.plot(points[key], points.AR, "o-", color=color, label=mode.replace("_", " "), linewidth=1.1)
                for point in points.itertuples():
                    prefix = {"beta":"B", "total_volatility":"T", "idiosyncratic_volatility":"I"}[mode]
                    axis.annotate(f"{prefix}{point.risk_penalty:g}", (getattr(point, key), point.AR), xytext=(4, 5 if mode != "total_volatility" else -11), textcoords="offset points", fontsize=8, color=color)
            zero = data.loc[data.risk_penalty == 0]
            axis.scatter(zero[key], zero.AR, marker="*", s=150, c="black", label="original alpha")
            if normalization_controls:
                control = normalization_controls[market]
                axis.scatter(control[key], control["AR"], marker="x", s=70, c="#686868", label="normalization only")
            axis.axhline(0, color="#dddddd", linewidth=.7)
            axis.grid(alpha=.2)
            axis.margins(x=.12, y=.18)
            axis.set_xlabel("Annualized volatility" if key == "STD" else "Maximum drawdown (higher is better)")
            axis.set_ylabel("Annualized net return")
            axis.set_title(market.upper())
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle(title+"\n2021-2022 purged validation; original baseline Top30 and costs", fontsize=12)
    fig.savefig(folder/f"historical_risk_{suffix}.png", dpi=180)
    svg = folder/f"historical_risk_{suffix}.svg"
    fig.savefig(svg)
    # Matplotlib path strings contain trailing spaces; keep exported XML tidy.
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines())+"\n")
    plt.close(fig)


def main():
    folder = ROOT/"research/records/20261008"
    table = pd.read_csv(folder/"historical_risk_validation_metrics.csv")
    assert len(table) == 20
    draw(table, "seed0", "Past-price risk scoring: alpha seed 0")
    seed_report = folder/"historical_risk_seed_validation_checks.json"
    if seed_report.exists():
        report = json.loads(seed_report.read_text())
        if report["completed_ensembles"] == report["planned_ensembles"] == 20:
            table = pd.read_csv(folder/"historical_risk_seed_validation_metrics.csv")
            controls = {row["market"]:row["normalization_only_control"]["validation"] for row in report["records"]}
            draw(table, "seeds012", "Past-price risk scoring: actual three-seed avg_none", controls)
    print("Saved completed-validation tradeoff figures")


if __name__ == "__main__":
    main()
