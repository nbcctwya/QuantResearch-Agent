"""Import genuinely completed immutable trials; preserve their original evidence."""
import json
from pathlib import Path

from research import ARTIFACTS
from research.common import config_id, digest_file, now, write_json
from research.historical_risk import verify_historical_source


def main():
    study = ARTIFACTS/"study"
    proof = json.loads((study/"historical_risk_full_trials.json").read_text())
    design = json.loads((study/"historical_risk_design.json").read_text())
    assert proof["complete"] and proof["completed"] == proof["planned"] == len(design["candidates"]) == 20
    assert proof["code"] == design["code"]
    imported = []
    for config, record in zip(design["candidates"], proof["trials"]):
        identifier = config_id(config)
        assert record["id"] == identifier and record["config"] == config
        source = Path(record["directory"])
        destination = ARTIFACTS/"trials"/identifier
        folder = source if source.exists() else destination
        for name, expected in [("result.json", record["result_sha256"]), ("run.json", record["run_sha256"]),
                               ("valid/predictions.pkl", record["prediction_sha256"]), ("components.json", record["components_sha256"])]:
            assert digest_file(folder/name) == expected
        result = json.loads((folder/"result.json").read_text())
        run = json.loads((folder/"run.json").read_text())
        assert result["config"] == config and result["status"] == "complete" and not result.get("smoke")
        assert run["code"] == design["code"] and run["source_package"] == str(Path(design["bundle"])/"research")
        verify_historical_source(config, folder)
        if source.exists() and source.resolve() != destination.resolve():
            assert not destination.exists(), "Coordinator already owns the target; do not overwrite"
            source.rename(destination)
        imported.append({"id":identifier, "original_directory":str(source), "directory":str(destination),
                         "original_started_at":run["started_at"], "original_completed_at":result["completed_at"],
                         "result_sha256":record["result_sha256"], "prediction_sha256":record["prediction_sha256"]})
    write_json(study/"historical_risk_import.json", {"created_at":now(), "imported":len(imported), "code":design["code"],
               "note":"Complete existing full runs moved without changing predictions, metrics, configs, source packages, start/completion times or run metadata.",
               "trials":imported})
    print({"actual_complete_runs_imported":len(imported), "test_accessed":False})


if __name__ == "__main__":
    main()
