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


# 重みを決める関数（task 2）


def make_store(responsibility_provider, *, with_responsibility_store=True):
    from federated_learning_experiments.learning.training.model_training_sample_store import (
        ModelTrainingSampleStore,
    )
    from federated_learning_experiments.learning.training.responsibility_weighted_training import (
        ResponsibilityWeightedTrainingStore,
    )

    store = ModelTrainingSampleStore()
    if with_responsibility_store:
        store.responsibility_training_store = ResponsibilityWeightedTrainingStore()
    store.training_responsibility_provider = responsibility_provider
    return store


def observed_samples(count):
    import torch

    from federated_learning_experiments.learning.training.model_training_sample_records import (
        ObservedTrainingSample,
    )

    return tuple(
        ObservedTrainingSample(
            input_features=torch.tensor([[float(i)]]), observed_class_labels=torch.tensor([[0.0]])
        )
        for i in range(count)
    )


def derive(store, source):
    from federated_learning_experiments.runtime.server_model_registration_and_aggregation import (
        derive_client_model_aggregation_weights,
    )

    return derive_client_model_aggregation_weights(
        training_sample_store=store, model_aggregation_weight_source=source
    )


def test_split_responsibilities_give_amounts_by_default_and_counts_when_selected():
    store = make_store(lambda model_id, sample: {3: 0.75, 5: 0.25})
    store.append_model_training_samples(model_id=3, training_samples=observed_samples(4))
    store.append_model_training_samples(model_id=5, training_samples=observed_samples(2))
    assert derive(store, AMOUNTS) == pytest.approx({3: 4.5, 5: 1.5})
    counts = derive(store, COUNTS)
    assert counts == {3: 4, 5: 2}
    assert all(type(value) is int for value in counts.values())


@pytest.mark.parametrize(
    "weight,probability",
    [(1.0, 0.3), (0.017, 0.9999), (3.7e-5, 1e-200), (0.5, 0.0), (0.0, 0.5)],
)
def test_one_model_responsibilities_make_amounts_exactly_equal_to_counts(weight, probability):
    from federated_learning_experiments.learning.training.responsibility_weighted_training import (
        calculate_training_responsibilities,
    )

    def provider(model_id, sample):
        return calculate_training_responsibilities(
            prediction_weights={model_id: weight},
            true_class_probabilities={model_id: probability},
            fallback_model_id=model_id,
        )

    store = make_store(provider)
    for _ in range(7):
        store.append_model_training_samples(model_id=2, training_samples=observed_samples(13))
    store.reassign_model_training_samples_id(original_model_id=2, reassigned_model_id=4)
    amounts, counts = derive(store, AMOUNTS), derive(store, COUNTS)
    assert counts == {4: 91}
    # 近似の比較を使わない（要求1.3）。
    assert amounts == counts
    assert all(type(value) is float for value in amounts.values())


@pytest.mark.parametrize("with_responsibility_store", [True, False])
def test_records_without_soft_assignment_give_counts_for_both_sources(with_responsibility_store):
    store = make_store(None, with_responsibility_store=with_responsibility_store)
    store.append_model_training_samples(model_id=1, training_samples=observed_samples(3))
    store.append_model_training_samples(model_id=0, training_samples=observed_samples(2))
    assert derive(store, AMOUNTS) == derive(store, COUNTS) == {1: 3, 0: 2}


@pytest.mark.parametrize("value", ["unknown", None, True, 0])
def test_invalid_source_is_rejected_before_reading_the_store(value):
    class UnreadableStore:
        def __getattr__(self, name):
            raise AssertionError(f"read {name}")

    with pytest.raises(ValueError):
        derive(UnreadableStore(), value)
