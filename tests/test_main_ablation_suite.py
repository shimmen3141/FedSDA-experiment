import os
import shutil
import subprocess

import pytest


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT = os.path.join(_REPO_ROOT, "tools", "run_main_ablation_suite.sh")
_BASH_UNAVAILABLE = os.name == "nt" or shutil.which("bash") is None


EXPECTED_VARIANTS = {
    "reference",
    "independent",
    "shared-backbone",
    "hard-routing",
    "global-routing",
    "meta-routing",
    "meta-switching-routing",
    "no-recalibration",
    "immediate-creation",
    "distance-average",
    "overall-esr",
    "class-adwin",
    "overall-adwin",
}


def test_main_ablation_suite_contains_reference_and_control_variants():
    with open(_SCRIPT, encoding="utf-8") as file:
        source = file.read()

    for variant in EXPECTED_VARIANTS:
        assert variant in source
    assert "FedSDA_NoCached_ResidualAdapter_ESR_RestartingSoftRouting" in source
    assert "FedSDA_NoCached_ResidualAdapter_ClassADWIN_RestartingSoftRouting" in source
    assert "FedSDA_NoCached_ResidualAdapter_ADWIN_RestartingSoftRouting" in source
    assert "--duplicate-policy error" in source
    assert "--workers" not in source
    assert "--out-dir" not in source
    assert "--raw-dir" not in source
    assert "--tag" not in source
    assert "--no-recovery" not in source


def test_main_ablation_suite_uses_direct_switching_as_reference():
    with open(_SCRIPT, encoding="utf-8") as file:
        source = file.read()

    start = source.index("final_soft_routing=(")
    end = source.index("\n)", start)
    reference_routing = source[start:end]

    assert "--soft-routing-context switching" in reference_routing
    assert "meta_switching" not in reference_routing
    assert "--soft-routing-top-combination" not in reference_routing
    assert "--soft-routing-meta-loss" not in reference_routing

    independent_start = source.index("        independent)")
    independent_end = source.index("            ;;", independent_start)
    independent_case = source[independent_start:independent_end]
    assert "--soft-routing-context switching" in independent_case
    assert "meta_switching" not in independent_case

    no_recalibration_start = source.index("        no-recalibration)")
    no_recalibration_end = source.index("            ;;", no_recalibration_start)
    no_recalibration_case = source[no_recalibration_start:no_recalibration_end]
    assert "--shared-backbone-routing-recalibration none" in no_recalibration_case
    assert "--soft-routing-context switching" in no_recalibration_case
    assert "meta_switching" not in no_recalibration_case


@pytest.mark.skipif(_BASH_UNAVAILABLE, reason="POSIX bashを利用できない環境")
def test_main_ablation_suite_lists_individual_variants():
    completed = subprocess.run(
        ["bash", _SCRIPT, "--list"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert set(completed.stdout.splitlines()) == EXPECTED_VARIANTS
