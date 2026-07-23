import pytest

from federated_drift_experiment import config
from federated_drift_experiment.data import (
    dataset_cli_choices,
    normalize_dataset_name,
)


def test_only_canonical_dataset_names_are_accepted():
    choices = dataset_cli_choices(config._FEATURE_DIMS)
    assert set(choices) == set(config._FEATURE_DIMS)
    assert "sea4" in choices
    assert "sea" not in choices


@pytest.mark.parametrize("legacy_name", ["sea", "circle", "sine"])
def test_legacy_dataset_names_are_rejected(legacy_name):
    with pytest.raises(ValueError, match="Unknown dataset"):
        normalize_dataset_name(legacy_name)
    with pytest.raises((KeyError, ValueError)):
        config.input_dim(legacy_name)


def test_canonical_dataset_name_is_unchanged():
    assert normalize_dataset_name("sea2") == "sea2"
