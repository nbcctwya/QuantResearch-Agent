"""Fixed full CUDA grid in separate directories; preserve coordinator-owned fits."""
import json
from pathlib import Path

from research import ARTIFACTS
from research.common import code_fingerprint, config_id, digest_file, now, write_json
from research.train import run_training


def main():
    study=ARTIFACTS/"study"
    design=json.loads((study/"volatility_target_design.json").read_text())
    assert code_fingerprint()==design["code"]
    assert Path(__import__("research").__file__).parent == Path(design["bundle"])/"research"
    for name in ["volatility_target_data_checks.json","volatility_target_gpu_checks.json"]:
        proof=json.loads((study/name).read_text());assert proof["passed"] and proof["code"]==design["code"]
    records=[]
    for config in design["candidates"]:
        identifier=config_id(config)
        folder=ARTIFACTS/"volatility_target_full_trials"/identifier
        assert not (ARTIFACTS/"trials"/identifier).exists(),"Do not duplicate a coordinator-owned fit"
        if not (folder/"result.json").exists():
            run_training(config,folder)
        result=json.loads((folder/"result.json").read_text())
        run=json.loads((folder/"run.json").read_text())
        assert result["config"]==config and result["status"]=="complete" and not result.get("smoke")
        assert run["status"]=="complete" and run["code"]==design["code"] and run["source_package"]==str(Path(design["bundle"])/"research")
        records.append({"id":identifier,"config":config,"directory":str(folder),
            "files":{name:digest_file(folder/name) for name in ["config.json","run.json","result.json","best.pt","last.pt",
                "volatility_target.json","target_transform.json","valid/predictions.pkl"]}})
        write_json(study/"volatility_target_full_trials.json",{"updated_at":now(),"complete":len(records)==8,
            "planned":8,"completed":len(records),"code":design["code"],"trials":records,
            "scope":"actual new full CUDA fits and baseline validation backtests; separate outputs until verified import; no test"})
        print({"id":identifier,"market":config["market"],"base":config["volatility_target_base"],
            "power":config["volatility_target_power"],"completed_full_fits":len(records)},flush=True)
    print({"complete_full_grid":8,"test_accessed":False},flush=True)


if __name__=="__main__":
    main()
