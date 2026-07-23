from federated_drift_experiment import config
from federated_drift_experiment.data import (
    dataset_cli_choices,
    normalize_dataset_in_text,
    normalize_dataset_name,
)


def test_legacy_dataset_names_map_to_paper_aligned_names():
    assert normalize_dataset_name("sea") == "sea4"
    assert normalize_dataset_name("circle") == "circle2"
    assert normalize_dataset_name("sine") == "sine2"
    assert normalize_dataset_name("sea2") == "sea2"


def test_legacy_names_remain_accepted_at_input_boundary():
    choices = dataset_cli_choices(config._FEATURE_DIMS)
    assert "circle2" in choices
    assert "circle" in choices
    assert config.input_dim("sea") == config.input_dim("sea4") == 3
    assert config.num_concepts("sine") == config.num_concepts("sine2") == 2


def test_dataset_tokens_in_historical_filenames_are_normalized():
    assert normalize_dataset_in_text(
        "FedDrift_v2_circle_seed0.npz"
    ) == "FedDrift_v2_circle2_seed0.npz"
    assert normalize_dataset_in_text("mnist2-sea2") == "mnist2-sea2"
