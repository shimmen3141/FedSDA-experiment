"""候補のepoch学習を実旧・設定契約・乱数へ対照する。"""

from dataclasses import FrozenInstanceError, fields

import pytest

from federated_learning_experiments.learning.training.candidate_epoch_training_settings import (
    CandidateEpochTrainingSettings,
)


@pytest.fixture
def valid_settings_arguments():
    return dict(
        candidate_training_strategy="validation_loss_early_stopping",
        maximum_epoch_count=5,
        maximum_batch_sample_count=3,
        validation_sample_fraction=0.2,
        consecutive_non_improving_epoch_limit=2,
        minimum_validation_loss_decrease=0.0001,
    )


@pytest.mark.parametrize(
    "invalid_settings_field_name,invalid_settings_field_value",
    [
        ("candidate_training_strategy", "early_stopping"),
        ("candidate_training_strategy", "fixed"),
        ("candidate_training_strategy", "none"),
        ("candidate_training_strategy", None),
        ("maximum_epoch_count", -1),
        ("maximum_epoch_count", True),
        ("maximum_epoch_count", 1.5),
        ("maximum_epoch_count", "3"),
        ("maximum_batch_sample_count", 0),
        ("maximum_batch_sample_count", -1),
        ("maximum_batch_sample_count", True),
        ("maximum_batch_sample_count", 1.5),
        ("validation_sample_fraction", 0),
        ("validation_sample_fraction", 1),
        ("validation_sample_fraction", float("inf")),
        ("validation_sample_fraction", float("nan")),
        ("validation_sample_fraction", True),
        ("validation_sample_fraction", "0.2"),
        ("consecutive_non_improving_epoch_limit", 0),
        ("consecutive_non_improving_epoch_limit", True),
        ("consecutive_non_improving_epoch_limit", 1.5),
        ("minimum_validation_loss_decrease", -0.01),
        ("minimum_validation_loss_decrease", float("inf")),
        ("minimum_validation_loss_decrease", float("nan")),
        ("minimum_validation_loss_decrease", True),
        ("minimum_validation_loss_decrease", "0"),
    ],
)
def test_candidate_epoch_training_settings_contract(
    valid_settings_arguments, invalid_settings_field_name, invalid_settings_field_value
):
    settings_argument_values = dict(valid_settings_arguments)
    settings_argument_values[invalid_settings_field_name] = invalid_settings_field_value
    with pytest.raises(ValueError):
        CandidateEpochTrainingSettings(**settings_argument_values)
    candidate_epoch_training_settings = CandidateEpochTrainingSettings(**valid_settings_arguments)
    assert tuple(
        settings_field.name for settings_field in fields(candidate_epoch_training_settings)
    ) == tuple(valid_settings_arguments)
    with pytest.raises(FrozenInstanceError):
        candidate_epoch_training_settings.maximum_epoch_count = 10
    with pytest.raises(TypeError):
        CandidateEpochTrainingSettings(*valid_settings_arguments.values())
    for configuration_parameter_name in valid_settings_arguments:
        settings_argument_values = dict(valid_settings_arguments)
        del settings_argument_values[configuration_parameter_name]
        with pytest.raises(TypeError):
            CandidateEpochTrainingSettings(**settings_argument_values)
    valid_settings_arguments["maximum_epoch_count"] = 0
    valid_settings_arguments["minimum_validation_loss_decrease"] = 0
    assert CandidateEpochTrainingSettings(**valid_settings_arguments).maximum_epoch_count == 0
    for candidate_training_strategy in (
        "fixed_epoch_training",
        "validation_loss_early_stopping",
        "skip_training",
    ):
        valid_settings_arguments["candidate_training_strategy"] = candidate_training_strategy
        assert (
            CandidateEpochTrainingSettings(**valid_settings_arguments).candidate_training_strategy
            == candidate_training_strategy
        )
