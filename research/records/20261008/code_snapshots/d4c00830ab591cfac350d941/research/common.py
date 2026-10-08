from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import ROOT


def now():
    return datetime.now(timezone.utc).isoformat()


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(clean_json(value), indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_id(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]


def code_fingerprint():
    directory = Path(__file__).resolve().parent
    return {"research/"+p.name: digest_file(p) for p in sorted(directory.glob("*.py"))}


def freeze_code(destination):
    """Create/reuse an immutable-by-convention package containing the launch-time source."""
    contents = {p.name:p.read_bytes() for p in sorted((ROOT/"research").glob("*.py"))}
    hashes = {"research/"+name:hashlib.sha256(data).hexdigest() for name,data in contents.items()}
    identifier = hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:24]
    bundle = Path(destination)/identifier
    if not bundle.exists():
        temporary = bundle.with_name(bundle.name+f".building.{os.getpid()}")
        (temporary/"research").mkdir(parents=True,exist_ok=False)
        for name,data in contents.items():
            (temporary/"research"/name).write_bytes(data)
        write_json(temporary/"manifest.json",{"created_at":now(),"workspace":str(ROOT),"files":hashes})
        try:
            temporary.rename(bundle)
        except FileExistsError:
            shutil.rmtree(temporary)
    for name,expected in hashes.items():
        if digest_file(bundle/name) != expected:
            raise ValueError("A frozen code bundle has changed")
    return bundle


def model_artifact_hashes(directory):
    directory = Path(directory)
    return {name:digest_file(directory/name) for name in ["best.pt","model.txt","model.npz","components.json"]
            if (directory/name).exists()}
