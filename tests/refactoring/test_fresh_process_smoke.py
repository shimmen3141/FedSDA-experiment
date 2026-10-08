"""新実装だけの流れ（fresh_process_smoke.py）を、別processで実行する。"""

import os
import subprocess
import sys
from pathlib import Path

SMOKE_SCRIPT_PATH = Path(__file__).with_name("fresh_process_smoke.py")


def test_fresh_process_smoke_runs_without_legacy_or_test_modules():
    # pytestのprocessが読み込んだmoduleやPYTHONPATHを引き継がない、新しいprocessで実行する。
    child_environment = {name: value for name, value in os.environ.items() if name != "PYTHONPATH"}
    completed_process = subprocess.run(
        [sys.executable, str(SMOKE_SCRIPT_PATH)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=SMOKE_SCRIPT_PATH.parents[2],
        env=child_environment,
    )
    assert completed_process.returncode == 0, completed_process.stdout + completed_process.stderr
    assert "FRESH PROCESS SMOKE PASSED" in completed_process.stdout
