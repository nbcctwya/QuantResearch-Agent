"""Actual individual and avg_none target metrics; validation period only."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np

from research import ARTIFACTS, ROOT
from research.common import digest_file, now, write_json
from research.selection import verify_confirmation_record


def main():
    folder=ROOT/"research/records/20261008"
    source=folder/"market_residual_validation_checks.json"
    report=json.loads(source.read_text())
    study=report["seed_study"]["actual_seed_ensembles"]
    assert study["completed"]==study["planned"]==8 and study["source_files_unchanged"]
    records=study["records"]
    for record in records:verify_confirmation_record(record)
    fig,axes=plt.subplots(2,4,figsize=(15,7),layout="constrained")
    metrics=[("RankIC","higher"),("AR","higher"),("STD","lower"),("MDD","higher")]
    labels=["Raw","CS excess","Beta 0.5","Beta 1.0"]
    individual_files={}
    for row,market in enumerate(["csi300","sp500"]):
        part=[record for record in records if record["config"]["market"]==market]
        assert len(part)==4
        for column,(metric,direction) in enumerate(metrics):
            axis=axes[row,column]
            for position,record in enumerate(part):
                individual=[]
                for source_record in record["sources"]:
                    path=ARTIFACTS/"trials"/source_record["id"]/"result.json"
                    assert digest_file(path)==source_record["files"]["result.json"]
                    individual_files[str(path)]=digest_file(path)
                    individual.append(json.loads(path.read_text())["validation"][metric])
                axis.scatter(position+np.array([-.12,0,.12]),individual,
                    color="#888888",s=22,alpha=.75,zorder=2)
                axis.scatter(position,record["validation"][metric],color="#1878aa",marker="D",s=48,zorder=3)
            axis.set_xticks(range(4),labels,rotation=20)
            axis.set_title(f"{market.upper()}  {metric} ({direction} is better)",fontsize=10)
            axis.grid(axis="y",alpha=.2)
            if metric!="RankIC":axis.yaxis.set_major_formatter(PercentFormatter(1))
    fig.suptitle("Training-target controls: purged 2021-2022 validation, unchanged baseline evaluation",fontsize=13)
    fig.legend(handles=[Line2D([],[],marker="o",linestyle="",color="#888888",label="Actual individual seeds 0 / 1 / 2"),
        Line2D([],[],marker="D",linestyle="",color="#1878aa",label="Actual avg_none forecast backtest")],
        loc="outside lower center",ncol=2,frameon=False)
    paths=[folder/"market_targets_seeds012.png",folder/"market_targets_seeds012.svg"]
    for path in paths:
        fig.savefig(path,dpi=170)
        if path.suffix==".svg":path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")
    plt.close(fig)
    assert all(digest_file(path)==value for path,value in individual_files.items())
    write_json(folder/"market_target_validation_plot.json",{"created_at":now(),"source":source.name,
        "source_sha256":digest_file(source),"actual_ensembles":8,"actual_individual_fits":24,
        "individual_result_files":individual_files,
        "files":{path.name:digest_file(path) for path in paths},
        "initial_harness_events":json.loads((ARTIFACTS/"study/market_target_plot_events.json").read_text())
            if (ARTIFACTS/"study/market_target_plot_events.json").exists() else [],
        "scope":"seed dots are individual validation metrics; diamonds recompute all metrics from averaged predictions, not averages of individual metrics; no holdout"})
    print({"actual_ensembles":8,"actual_individual_fits":24,"plots":[path.name for path in paths]})


if __name__=="__main__":main()
