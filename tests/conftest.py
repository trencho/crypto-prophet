"""Shared test setup for crypto-prophet.

Two things must happen *before* any ``src`` module (or ``definitions``) is
imported, so they run at module import time here, above every other import:

1. ``VOLUME_PATH`` is redirected to a throwaway temp dir. ``definitions.py``
   resolves every data/model/result path from it at import time, so setting it
   first keeps the whole suite off the real ``data/``, ``models/`` and
   ``results/`` trees.
2. ``<repo>/src`` and the repo root are put on ``sys.path`` so the app's
   ``from api...`` / ``from modeling...`` / ``from definitions`` imports resolve.
"""

import os
import sys
import tempfile
from pathlib import Path

# --- path/env bootstrap (must precede any `src`/`definitions` import) ---------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"

# A session-scoped temp dir that stands in for the mounted data volume.
_VOLUME = Path(tempfile.mkdtemp(prefix="crypto-prophet-tests-"))
os.environ["VOLUME_PATH"] = str(_VOLUME)

for _p in (str(_SRC), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Safe to import now that VOLUME_PATH + sys.path are set.
import definitions  # noqa: E402

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def volume_path() -> Path:
    """The temp dir standing in for VOLUME_PATH (== all definitions.* roots)."""
    return _VOLUME


@pytest.fixture
def results_errors_dir():
    """Create and return ``RESULTS_ERRORS_PATH/data/<coin>/<model>`` for a case.

    ``save_errors`` writes with ``DataFrame.to_csv`` (which does not create
    parent dirs), so the leaf must exist first.
    """

    def _make(coin_symbol: str, model_name: str) -> Path:
        leaf = Path(definitions.RESULTS_ERRORS_PATH) / "data" / coin_symbol / model_name
        leaf.mkdir(parents=True, exist_ok=True)
        return leaf

    return _make


@pytest.fixture
def models_dir():
    """Create and return ``MODELS_PATH/<ClassName>`` for save/load round-trips.

    ``BaseRegressionModel.save`` writes to ``<file_path>/<ClassName>/<ClassName>.pkl``
    and does not create parent dirs, so the class subdir must exist first.
    """

    def _make(class_name: str) -> Path:
        base = Path(definitions.MODELS_PATH)
        (base / class_name).mkdir(parents=True, exist_ok=True)
        return base

    return _make
