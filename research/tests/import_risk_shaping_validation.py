"""Import all completed grid runs after the seed follower finishes its file reads."""
import json
from pathlib import Path

from research import ARTIFACTS
from research.common import code_fingerprint, config_id, digest_file, now, write_json
from research.historical_risk import verify_historical_source


def main():
    study = ARTIFACTS/"study"
    design = json.loads((study/"risk_shaping_design.json").read_text())
    proof = json.loads((study/"risk_shaping_full_trials.json").read_text())
    seeds = json.loads((study/"risk_shaping_seed_validation_checks.json").read_text())
    activity = json.loads((study/"risk_shaping_ensemble_follower.json").read_text())
    assert code_fingerprint() == design["code"]
    assert Path(__import__("research").__file__).parent == Path(design["bundle"])/"research"
    assert proof["complete"] and proof["completed"] == proof["planned"] == len(design["candidates"]) == 38
    assert proof["code"] == seeds["code"] == design["code"]
    assert seeds["completed"] == seeds["planned"] == 38 and seeds["source_files_unchanged"]
    assert activity["phase"] == "complete", "Do not move files while the seed follower is still reading them"
    imported = []
    for config, row in zip(design["candidates"], proof["trials"]):
        identifier = config_id(config)
        assert row["id"] == identifier and row["config"] == config
        original = Path(row["directory"])
        destination = ARTIFACTS/"trials"/identifier
        folder = original if original.exists() else destination
        for name, expected in row["files"].items():
            assert digest_file(folder/name) == expected
        result = json.loads((folder/"result.json").read_text())
        run = json.loads((folder/"run.json").read_text())
        assert result["config"] == config and result["status"] == "complete" and not result.get("smoke")
        assert run["code"] == design["code"] and run["source_package"] == str(Path(design["bundle"])/"research")
        verify_historical_source(config, folder)
        if original.exists() and original.resolve() != destination.resolve():
            assert not destination.exists(), "Never overwrite a coordinator-owned trial"
            original.rename(destination)
        imported.append({"id":identifier, "original_directory":str(original),"directory":str(destination),
            "original_started_at":run["started_at"], "original_completed_at":result["completed_at"], "files":row["files"]})
    write_json(study/"risk_shaping_import.json", {"created_at":now(),"imported":len(imported),"code":design["code"],
        "scope":"genuine complete validation outputs moved without changing predictions, metrics, metadata or timestamps; test sealed", "trials":imported})
    print({"genuine_completed_trials_imported":len(imported),"completed_seed_diagnostics":38,"test_accessed":False})


if __name__ == "__main__":
    main()
