"""Reproducible, unattended research using the pinned baseline protocol."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT / "kbs-exp"
ARTIFACTS = ROOT / "research" / "artifacts"
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))
