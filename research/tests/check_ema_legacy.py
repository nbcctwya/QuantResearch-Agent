"""Exact disabled-EMA GPU compatibility against the prior frozen training package."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import torch

from research import ARTIFACTS, ROOT
from research.common import code_fingerprint, config_id, now, write_json


def main():
    study = ARTIFACTS / "study"
    plan = json.loads((study / "ema_smoke_plan.json").read_text())[:2]
    old = ARTIFACTS / "code_releases/bd60504de52434796f35fd4c"
    environment = dict(os.environ, KBS_RESEARCH_WORKSPACE_ROOT=str(ROOT),
                       OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    records = []
    for item in plan:
        config = json.loads(Path(item["config"]).read_text())
        config.pop("ema_decay")
        outputs = []
        for name, bundle in [("old", old), ("current", Path(item["bundle"]))]:
            output = ARTIFACTS / "smokes" / f"ema_disabled_{config_id(config)}_{name}"
            output.mkdir(exist_ok=False)
            write_json(output / "config.json", config)
            with (output / "console.log").open("w") as log:
                subprocess.run([sys.executable, "-u", "-m", "research.train", "--config", str(output / "config.json"),
                                "--out", str(output)], cwd=bundle, env=environment,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            outputs.append(output)
        for name in ["best.pt", "last.pt"]:
            expected = torch.load(outputs[0] / name, map_location="cpu", weights_only=False)
            actual = torch.load(outputs[1] / name, map_location="cpu", weights_only=False)
            assert "ema" not in actual
            assert expected["epoch"] == actual["epoch"]
            for key, value in expected["model"].items():
                torch.testing.assert_close(value, actual["model"][key], atol=0, rtol=0)
        before = [json.loads(line) for line in (outputs[0] / "epochs.jsonl").read_text().splitlines()]
        after = [json.loads(line) for line in (outputs[1] / "epochs.jsonl").read_text().splitlines()]
        for first, second in zip(before, after):
            for key in ["epoch", "train_loss", "valid_selection_score", "best_selection_score", "valid", "optimizer_steps"]:
                assert first[key] == second[key], (config["market"], key)
        assert len(before) == len(after) == 3
        pd.testing.assert_frame_equal(pd.read_pickle(outputs[0] / "valid/predictions.pkl"),
                                      pd.read_pickle(outputs[1] / "valid/predictions.pkl"), check_exact=True)
        records.append({"market": config["market"], "config": config, "old_bundle": str(old),
                        "current_bundle": item["bundle"], "exact_training_losses_and_validation": True,
                        "exact_raw_parameters_and_best_epoch": True, "exact_validation_predictions": True,
                        "scope": "three-epoch real GPU smoke; excluded from full trials"})
        write_json(study / "ema_legacy_gpu_checks.json", {"created_at": now(), "passed": len(records) == len(plan),
                   "markets_completed": len(records), "records": records, "code": code_fingerprint()})
        print({"market": config["market"], "disabled_ema_legacy_exact": True}, flush=True)


if __name__ == "__main__":
    main()
