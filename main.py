"""Container / production entrypoint (referenced by docker-compose as ``main:app``).

The FastAPI app lives at ``src/api/app.py`` and its modules import as ``from api...``
and ``from definitions...``, so both the repo root and ``src/`` must be importable.
Running ``gunicorn main:app`` from the repo root puts the root on ``sys.path`` (for
``definitions``); this module adds ``src/`` (for the ``api`` package) before importing.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from api.app import app  # noqa: E402

__all__ = ["app"]
