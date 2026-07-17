import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from recovery_analysis import _method_name, _recovery_label


def test_recovery_label_preserves_method_version():
    assert _method_name("FedSDA_v3 δ_adwin sweep (γ=0.1) [0.05]") == "FedSDA_v3"
    assert _recovery_label(
        "FedSDA_v2 δ_adwin sweep (γ=0.1) [0.05]", "δ_adwin"
    ) == "FedSDA_v2 (δ_adwin=0.05)"
    assert _recovery_label(
        "FedDrift_v2 batch sweep (δ=0.1) [50]", "batch"
    ) == "FedDrift_v2 (batch=50)"
