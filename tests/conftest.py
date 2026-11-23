import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.pop("OPENROUTER_API_KEY", None)      # the tests never call a model


@pytest.fixture(scope="session")
def prepared():
    from hpo_severity import paths
    paths.prepare()
    return paths
