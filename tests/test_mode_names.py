from federated_drift_experiment.experiment import MODE_SPECS
from federated_drift_experiment.mode_names import (
    BASELINE_MODES,
    FEDDRIFT_MODES,
    FEDSDA_MODES,
    fedsda_detector_name,
    is_adwin_mode,
    is_esr_mode,
)


def test_mode_registry_contains_only_current_public_names():
    assert set(MODE_SPECS) == set(FEDSDA_MODES + FEDDRIFT_MODES + BASELINE_MODES)
    assert "FedSDA" not in MODE_SPECS
    assert "FedDrift_v2" not in MODE_SPECS
    assert not any("_v2" in mode or "_v3" in mode for mode in MODE_SPECS)


def test_legacy_mode_names_are_not_public():
    for legacy_name in ("FedSDA", "FedSDA_v2", "FedSDA_v3.3", "FedDrift_v2"):
        assert legacy_name not in MODE_SPECS


def test_detector_family_is_parsed_for_overall_and_class_modes():
    assert fedsda_detector_name("FedSDA_NoCached_ADWIN") == "ADWIN"
    assert fedsda_detector_name("FedSDA_NoCached_ClassADWIN") == "ClassADWIN"
    assert fedsda_detector_name("FedSDA_Cached_ClassESR") == "ClassESR"
    assert is_adwin_mode("FedSDA_NoCached_ClassADWIN")
    assert is_esr_mode("FedSDA_Cached_ClassESR")
