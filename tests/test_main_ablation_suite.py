import os
import shutil
import subprocess

import pytest


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT = os.path.join(_REPO_ROOT, "tools", "run_main_ablation_suite.sh")
_BASH_UNAVAILABLE = os.name == "nt" or shutil.which("bash") is None


EXPECTED_VARIANTS = {
    "independent",
    "shared-backbone",
    "hard-routing",
    "global-routing",
    "switching-routing",
    "meta-routing",
    "no-recalibration",
    "immediate-creation",
    "distance-average",
    "overall-esr",
    "overall-adwin",
}


def test_main_ablation_suite_contains_only_missing_control_variants():
    with open(_SCRIPT, encoding="utf-8") as file:
        source = file.read()

    for variant in EXPECTED_VARIANTS:
        assert variant in source
    assert "FedSDA_NoCached_ResidualAdapter_ESR_RestartingSoftRouting" in source
    assert "FedSDA_NoCached_ResidualAdapter_ADWIN_RestartingSoftRouting" in source
    assert "--duplicate-policy error" in source
    assert "--workers" not in source
    assert "--out-dir" not in source
    assert "--raw-dir" not in source
    assert "--tag" not in source
    assert "--no-recovery" not in source


@pytest.mark.skipif(_BASH_UNAVAILABLE, reason="POSIX bashを利用できない環境")
def test_main_ablation_suite_lists_individual_variants():
    completed = subprocess.run(
        ["bash", _SCRIPT, "--list"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert set(completed.stdout.splitlines()) == EXPECTED_VARIANTS
