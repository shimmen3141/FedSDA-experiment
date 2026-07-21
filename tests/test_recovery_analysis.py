import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from recovery_analysis import (
    _method_name,
    _recovery_label,
    _representative_labels,
    plot_recovery,
)


def test_recovery_label_preserves_method_version():
    assert _method_name("FedSDA_Cached_ADWIN δ_ADWIN sweep (A=50, γ=0.1) [0.05]") == "FedSDA_Cached_ADWIN"
    assert _recovery_label(
        "FedSDA_NoCached_ADWIN δ_ADWIN sweep (A=50, γ=0.1) [0.05]", "δ_ADWIN"
    ) == "FedSDA_NoCached_ADWIN (δ_ADWIN=0.05)"
    assert _recovery_label(
        "FedDrift B_detect sweep (δ_FedDrift=0.1) [50]", "B_detect"
    ) == "FedDrift (B_detect=50)"


def test_representative_labels_select_one_default_config_per_version():
    labels = [
        "FedSDA_NoCached_ClassADWIN A sweep (δ_ADWIN=0.05, γ=0.1) [25]",
        "FedSDA_NoCached_ClassADWIN A sweep (δ_ADWIN=0.05, γ=0.1) [50]",
        "FedSDA_NoCached_ClassADWIN δ_ADWIN sweep (A=50, γ=0.1) [0.05]",
        "FedSDA_Cached_ClassADWIN A sweep (δ_ADWIN=0.05, γ=0.1) [50]",
    ]
    selected = _representative_labels(
        labels, {"A": 50, "δ_ADWIN": 0.05}
    )

    assert len(selected) == 2
    assert all(" A sweep" in label and "[50]" in label for label in selected)
    assert {label.split(maxsplit=1)[0] for label in selected} == {
        "FedSDA_NoCached_ClassADWIN", "FedSDA_Cached_ClassADWIN"
    }


def test_plot_recovery_skips_file_when_no_series(tmp_path):
    output = tmp_path / "empty.png"
    agg = {
        ("circle", "FedSDA_NoCached_ClassADWIN A sweep [50]"): {
            "mean": [1.0], "std": [0.0], "n_drifts": 1, "n_seeds": 1,
        }
    }

    generated = plot_recovery(
        agg, 0, output, 0, "empty", label_filter=lambda _: False
    )

    assert generated is False
    assert not output.exists()
