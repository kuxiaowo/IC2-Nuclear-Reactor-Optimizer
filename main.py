"""Run the IC2 reactor optimizer directly from a source checkout."""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_ROOT = PROJECT_ROOT / "backend"


def main() -> None:
    # Keep imports, static frontend discovery and trace storage independent of
    # the directory from which ``python path/to/main.py`` was invoked.
    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, str(BACKEND_ROOT))

    from ic2_reactor.__main__ import main as run_server

    run_server()


if __name__ == "__main__":
    main()
