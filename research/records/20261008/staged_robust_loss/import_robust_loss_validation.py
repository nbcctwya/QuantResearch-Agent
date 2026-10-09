"""Import genuine full neural outputs without rewriting their recorded evidence."""
import json
from pathlib import Path

from research import ARTIFACTS, ROOT
from research.common import code_fingerprint, config_id, digest_file, now, write_json


def main():
    study=ARTIFACTS/"study"
    design=json.loads((study/"robust_loss_design.json").read_text())
    proof=json.loads((study/"robust_loss_full_trials.json").read_text())
    assert code_fingerprint()==design["code"]==proof["code"]
    assert {"research/"+path.name:digest_file(path) for path in sorted((ROOT/"research").glob("*.py"))}==design["code"],"Install validated robust source in the coordinator before exposing robust configs to adaptation"
    assert Path(__import__("research").__file__).parent==Path(design["bundle"])/"research"
    assert proof["complete"] and proof["planned"]==proof["completed"]==len(design["candidates"])==12
    imported=[]
    for config,row in zip(design["candidates"],proof["trials"]):
        identifier=config_id(config);assert identifier==row["id"] and config==row["config"]
        original=Path(row["directory"]);destination=ARTIFACTS/"trials"/identifier
        folder=original if original.exists() else destination
        timestamps={name:(folder/name).stat().st_mtime_ns for name in row["files"]}
        for name,expected in row["files"].items():assert digest_file(folder/name)==expected
        run=json.loads((folder/"run.json").read_text());result=json.loads((folder/"result.json").read_text())
        assert run["status"]==result["status"]=="complete" and result["config"]==config and not result.get("smoke")
        assert run["code"]==design["code"] and run["source_package"]==str(Path(design["bundle"])/"research")
        if original.exists() and original.resolve()!=destination.resolve():
            assert not destination.exists(),"Never overwrite a coordinator-owned trial"
            original.rename(destination)
        assert all(digest_file(destination/name)==expected for name,expected in row["files"].items())
        assert timestamps=={name:(destination/name).stat().st_mtime_ns for name in row["files"]}
        imported.append({"id":identifier,"original_directory":str(original),"directory":str(destination),
            "original_started_at":run["started_at"],"original_completed_at":result["completed_at"],"files":row["files"],
            "file_mtimes_ns":timestamps,"bytes_and_mtimes_unchanged":True})
    write_json(study/"robust_loss_import.json",{"created_at":now(),"imported":12,"code":design["code"],
        "scope":"genuine complete neural validation outputs moved with unchanged bytes and file timestamps; no test","trials":imported})
    print({"genuine_complete_neural_trials_imported":12,"test_accessed":False})


if __name__=="__main__":
    main()
