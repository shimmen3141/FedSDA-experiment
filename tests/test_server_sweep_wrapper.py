import os
import shutil
import subprocess

import pytest


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT = os.path.join(_REPO_ROOT, "tools", "run_server_sweep.sh")
_BASH_UNAVAILABLE = os.name == "nt" or shutil.which("bash") is None


def test_server_sweep_wrapper_exposes_configurable_runtime_defaults():
    with open(_SCRIPT, encoding="utf-8") as file:
        source = file.read()

    for variable in (
        "FDE_RUN_DIR", "FDE_WORKERS", "FDE_VENV_DIR", "FDE_PYTHON",
        "FDE_TIME_BIN", "FDE_NO_RECOVERY", "FDE_TAG", "FDE_DRY_RUN",
    ):
        assert variable in source
    assert "set -euo pipefail" in source


@pytest.mark.skipif(_BASH_UNAVAILABLE, reason="POSIX bashが利用できない環境")
def test_server_sweep_wrapper_resolves_runtime_options(tmp_path):
    env = dict(os.environ)
    env.update({
        "FDE_RUN_DIR": str(tmp_path),
        "FDE_WORKERS": "3",
        "FDE_PYTHON": "/usr/bin/python3",
        "FDE_DRY_RUN": "1",
    })

    completed = subprocess.run(
        [
            "bash", _SCRIPT, "example",
            "--datasets", "circle2", "--no-feddrift", "--no-baselines",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "Workers: 3" in completed.stdout
    assert "--workers 3" in completed.stdout
    assert "example/pareto" in completed.stdout.replace("\\", "/")
    assert "--no-recovery" in completed.stdout


@pytest.mark.skipif(_BASH_UNAVAILABLE, reason="POSIX bashが利用できない環境")
def test_server_sweep_wrapper_rejects_owned_options(tmp_path):
    env = dict(os.environ)
    env.update({"FDE_RUN_DIR": str(tmp_path), "FDE_DRY_RUN": "1"})

    completed = subprocess.run(
        ["bash", _SCRIPT, "example", "--workers", "2"],
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 2
    assert "managed by run_server_sweep.sh" in completed.stderr
