"""Reproducible, unattended research using the pinned baseline protocol."""

import sys
import os
from pathlib import Path

ROOT = Path(os.environ.get("KBS_RESEARCH_WORKSPACE_ROOT",Path(__file__).resolve().parents[1])).resolve()
EXPERIMENT_ROOT = ROOT / "kbs-exp"
ARTIFACTS = ROOT / "research" / "artifacts"
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))
