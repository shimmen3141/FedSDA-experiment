from federated_drift_experiment.experiment import MODE_SPECS
from federated_drift_experiment.mode_names import (
    BASELINE_MODES,
    FEDDRIFT_MODES,
    FEDSDA_MODES,
    normalize_legacy_mode,
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
