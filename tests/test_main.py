from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_source_entrypoint_registers_canonical_package_for_spawn_children():
    script = (
        "import runpy, sys; "
        f"runpy.run_path({str(PROJECT_ROOT / 'main.py')!r}, run_name='__mp_main__'); "
        "import ic2_reactor; "
        "assert ic2_reactor.__name__ == 'ic2_reactor'; "
        "assert 'backend.ic2_reactor' not in sys.modules"
    )

    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=PROJECT_ROOT.parent,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
