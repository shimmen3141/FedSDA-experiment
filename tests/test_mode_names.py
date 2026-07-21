from federated_drift_experiment.experiment import MODE_SPECS
from federated_drift_experiment.mode_names import (
    BASELINE_MODES,
    FEDDRIFT_MODES,
    FEDSDA_MODES,
    fedsda_detector_name,
    is_adwin_mode,
    is_esr_mode,
    normalize_legacy_mode,
    normalize_series_notation,
)


def test_mode_registry_contains_only_current_public_names():
    assert set(MODE_SPECS) == set(FEDSDA_MODES + FEDDRIFT_MODES + BASELINE_MODES)
    assert "FedSDA" not in MODE_SPECS
    assert "FedDrift_v2" not in MODE_SPECS
    assert not any("_v2" in mode or "_v3" in mode for mode in MODE_SPECS)


def test_legacy_result_names_map_to_current_names():
    assert normalize_legacy_mode("FedSDA_v2") == "FedSDA_NoCached_ADWIN"
    assert normalize_legacy_mode("FedSDA_v3.3") == "FedSDA_Cached_ClassESR"
    assert normalize_legacy_mode("FedDrift_v2") == "FedDrift"
    assert normalize_legacy_mode("FedSDA") == "FedSDA_Legacy"


def test_detector_family_is_parsed_for_overall_and_class_modes():
    assert fedsda_detector_name("FedSDA_NoCached_ADWIN") == "ADWIN"
    assert fedsda_detector_name("FedSDA_NoCached_ClassADWIN") == "ClassADWIN"
    assert fedsda_detector_name("FedSDA_Cached_ClassESR") == "ClassESR"
    assert is_adwin_mode("FedSDA_NoCached_ClassADWIN")
    assert is_esr_mode("FedSDA_Cached_ClassESR")


def test_old_sweep_notation_maps_to_current_plot_symbols():
    assert normalize_series_notation(
        "FedSDA_NoCached_ADWIN AGG_INTERVAL sweep (δ_adwin=0.05)"
    ) == "FedSDA_NoCached_ADWIN A sweep (δ_ADWIN=0.05)"
    assert normalize_series_notation(
        "FedDrift batch sweep (δ=0.1)"
    ) == "FedDrift B_detect sweep (δ_FedDrift=0.1)"
    assert normalize_series_notation(
        "FedDrift δ sweep (batch=50)"
    ) == "FedDrift δ_FedDrift sweep (B_detect=50)"
