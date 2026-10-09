"""Show all fixed scoring variants on actual three-seed baseline backtests."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from research import ROOT
from research.common import digest_file, now, write_json


def main():
    folder = ROOT/"research/records/20261008"
    report = json.loads((folder/"risk_shaping_validation_checks.json").read_text())
    assert report["three_seed_diagnostics"]["completed"] == 38
    assert report["three_seed_diagnostics"]["source_files_unchanged"]
    path = folder/"risk_shaping_seed_validation_metrics.csv"
    current = pd.read_csv(path)
    assert len(current) == 38
    old_path = folder/"historical_risk_seed_validation_metrics.csv"
    old = pd.read_csv(old_path).rename(columns={"risk_penalty":"penalty"})
    old = old.loc[old.risk_mode.isin(["beta","total_volatility"]) & (old.penalty>0)].copy()
    old["alpha_norm"], old["risk_transform"] = "cs_z", "linear"
    table = pd.concat([current,old],ignore_index=True)
    legacy = json.loads((folder/"historical_risk_seed_validation_checks.json").read_text())
    normalization = {row["market"]:row["normalization_only_control"]["validation"]
                     for row in legacy["records"]}
    fig, axes = plt.subplots(2,2,figsize=(12,8),layout="constrained")
    variants = [("cs_z","linear","#686868","--"), ("native","linear","#2176ae","-"),
                ("cs_z","positive","#c45a21","-"), ("native","positive","#44803f","-")]
    for row,market in enumerate(["csi300","sp500"]):
        zero = current.loc[(current.market==market)&(current.penalty==0)].iloc[0]
        for column,mode in enumerate(["beta","total_volatility"]):
            axis = axes[row,column]
            for scale,transform,color,style in variants:
                part = table.loc[(table.market==market)&(table.risk_mode==mode)&(table.penalty>0)
                    &(table.alpha_norm==scale)&(table.risk_transform==transform)].sort_values("penalty")
                assert len(part) == 3
                axis.plot(part.STD,part.AR,"o",linestyle=style,color=color,linewidth=1.2,
                          label=f"{scale} / {transform}")
                for point in part.itertuples():
                    axis.annotate(f"{point.penalty:g}",(point.STD,point.AR),
                        xytext=(3,4),textcoords="offset points",fontsize=7,color=color)
            axis.scatter([zero.STD],[zero.AR],marker="*",s=140,color="black",label="original alpha")
            normalized = normalization[market]
            axis.scatter([normalized["STD"]],[normalized["AR"]],marker="x",s=65,
                color="#8b6aab",label="normalization only")
            axis.axhline(0,color="#dddddd",linewidth=.7)
            axis.grid(alpha=.2)
            axis.set_xlabel("Annualized volatility (lower is better)")
            axis.set_ylabel("Annualized net return")
            axis.set_title(f"{market.upper()} / {mode.replace('_',' ')}")
            axis.margins(x=.15,y=.18)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=3,frameon=False)
    fig.suptitle("Actual three-seed avg_none: native scales and positive risk penalties\n"
        "2021-2022 purged validation; unchanged baseline Top30/drop5 and fees; labels show penalty",fontsize=12)
    matplotlib.rcParams["svg.hashsalt"]="kbs-risk-shaping-validation"
    png,svg=folder/"risk_shaping_seeds012.png",folder/"risk_shaping_seeds012.svg"
    fig.savefig(png,dpi=180)
    fig.savefig(svg,metadata={"Date":None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines())+"\n")
    plt.close(fig)
    write_json(folder/"risk_shaping_validation_plot.json",{"created_at":now(),
        "scope":"complete fixed three-seed validation grid and original matched scoring controls; no test",
        "files":{str(path):digest_file(path) for path in [path,old_path,png,svg]},
        "new_grid_rows":38,"old_nonzero_controls":len(old)})
    print({"new_grid_rows":38,"old_controls":len(old),"figure":str(png)})


if __name__ == "__main__":
    main()
