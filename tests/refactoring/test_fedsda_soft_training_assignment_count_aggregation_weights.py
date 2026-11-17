"""構成`fedsda-soft-training-assignment-count-aggregation-weights`（柔らかい学習の集約の重みと参加の判定だけを、
固い割り当ての標本の件数に替えた構成）を、設定・重みの関数・run・掃引の順に確かめる。"""

from dataclasses import replace

import pytest

from federated_learning_experiments.core.configuration_errors import RunSettingsValidationError
from federated_learning_experiments.core.settings_serialization import (
    convert_settings_to_plain_mapping,
)

NAME = "fedsda-soft-training-assignment-count-aggregation-weights"
COUNTS = "single_model_assignment_sample_counts"
AMOUNTS = "training_assignment_amounts"


def build_settings(name, dataset="sine2", count=300, client_count=2, interval=50):
    from federated_learning_experiments.configuration.experiment_run_conditions import (
        ExperimentRunConditions,
    )
    from federated_learning_experiments.sweep.experiment_configurations import (
        build_experiment_run_settings,
    )

    return build_experiment_run_settings(
        configuration_name=name,
        experiment_run_conditions=ExperimentRunConditions(
            dataset_name=dataset,
            random_seed=0,
            client_count=client_count,
            per_client_sample_count=count,
            server_aggregation_interval_per_client_samples=interval,
        ),
    )


def client_settings(name):
    return build_settings(name).run_participant_settings.run_client_settings


# 設定（task 1）


def test_default_source_is_current_amounts_and_is_not_saved():
    for name in ("fedsda", "fedsda-soft-training", "fedsda-training-mass-diagnostics"):
        settings = client_settings(name)
        assert settings.model_aggregation_weight_source == AMOUNTS
        assert "model_aggregation_weight_source" not in convert_settings_to_plain_mapping(settings)


def test_counts_are_accepted_with_soft_training_and_saved():
    soft = client_settings("fedsda-soft-training")
    changed = replace(soft, model_aggregation_weight_source=COUNTS)
    assert changed.model_aggregation_weight_source == COUNTS
    assert convert_settings_to_plain_mapping(changed)["model_aggregation_weight_source"] == COUNTS


@pytest.mark.parametrize("value", ["unknown", True, None, 1])
def test_invalid_source_values_are_rejected(value):
    with pytest.raises(RunSettingsValidationError):
        replace(client_settings("fedsda-soft-training"), model_aggregation_weight_source=value)


@pytest.mark.parametrize("name", ["fedsda", "fedsda-training-mass-diagnostics"])
def test_counts_are_rejected_without_soft_training(name):
    with pytest.raises(RunSettingsValidationError):
        replace(client_settings(name), model_aggregation_weight_source=COUNTS)


def test_client_exposes_the_configured_source():
    from federated_learning_experiments.runtime.fedsda_run_client import FedsdaRunClient

    assert isinstance(FedsdaRunClient.model_aggregation_weight_source, property)
    assert FedsdaRunClient.model_aggregation_weight_source.fset is None
    client = FedsdaRunClient.__new__(FedsdaRunClient)
    client._run_client_settings = replace(
        client_settings("fedsda-soft-training"), model_aggregation_weight_source=COUNTS
    )
    assert client.model_aggregation_weight_source == COUNTS
