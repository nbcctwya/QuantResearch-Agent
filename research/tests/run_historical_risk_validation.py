"""Run the fixed CPU score grid without colliding with the GPU coordinator."""
import json
from pathlib import Path
import traceback

from research import ARTIFACTS
from research.common import code_fingerprint, config_id, digest_file, now, write_json
from research.historical_risk import verify_historical_source
from research.train import run_training


def main():
    study = ARTIFACTS/"study"
    design = json.loads((study/"historical_risk_design.json").read_text())
    assert code_fingerprint() == design["code"]
    assert Path(__import__("research").__file__).parent == Path(design["bundle"])/"research"
    for name in ["historical_risk_cpu_checks", "historical_risk_route_checks"]:
        proof = json.loads((study/(name+".json")).read_text())
        assert proof["passed"] and proof["code"] == design["code"]
    records = []
    for config in design["candidates"]:
        identifier = config_id(config)
        existing = ARTIFACTS/"trials"/identifier
        destination = existing if (existing/"result.json").exists() else ARTIFACTS/"historical_risk_full_trials"/identifier
        try:
            if not (destination/"result.json").exists():
                run_training(config, destination)
            result = json.loads((destination/"result.json").read_text())
            run = json.loads((destination/"run.json").read_text())
            assert result["config"] == config and result["status"] == "complete" and not result.get("smoke")
            assert run["code"] == design["code"] and run["source_package"] == str(Path(design["bundle"])/"research")
            verify_historical_source(config, destination)
            records.append({"id":identifier, "config":config, "directory":str(destination),
                            "result_sha256":digest_file(destination/"result.json"),
                            "run_sha256":digest_file(destination/"run.json"),
                            "prediction_sha256":digest_file(destination/"valid/predictions.pkl"),
                            "components_sha256":digest_file(destination/"components.json")})
            write_json(study/"historical_risk_full_trials.json", {"created_at":now(), "complete":len(records)==len(design["candidates"]),
                       "completed":len(records), "planned":len(design["candidates"]), "code":code_fingerprint(), "trials":records,
                       "scope":"actual complete full validation runs, no test; temporary output paths prevent coordinator collisions"})
        except Exception:
            write_json(destination/"failure.json", {"at":now(), "config":config, "traceback":traceback.format_exc()})
            raise


if __name__ == "__main__":
    main()
