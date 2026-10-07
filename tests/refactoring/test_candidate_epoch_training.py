"""候補のepoch学習を実旧・設定契約・乱数へ対照する。"""

from copy import deepcopy
from dataclasses import FrozenInstanceError, fields
from unittest.mock import patch

import pytest
import torch
from test_candidate_classifier_construction import (
    build_candidate_construction_oracle,
    create_legacy_candidate,
)
from test_joint_model_parameter_update import assert_nested_state_equal
from test_shared_feature_extractor_attachment import (
    assert_parameter_values_and_gradients_unchanged,
    snapshot_parameter_values_and_gradients,
)

from federated_drift_experiment import config
from federated_drift_experiment.clients.base import BaseClient
from federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation import (
    evaluate_classifier_per_sample_bounded_losses,
)
from federated_learning_experiments.learning.training.candidate_epoch_training import (
    CandidateEpochTrainingResult,
    _train_candidate_dataset_epochs,
    _validate_candidate_epoch_training_inputs,
    train_candidate_classifier_epochs,
)
from federated_learning_experiments.learning.training.candidate_epoch_training_settings import (
    CandidateEpochTrainingSettings,
)
from federated_learning_experiments.learning.training.joint_model_parameter_update import (
    perform_joint_model_parameter_update,
)
from federated_learning_experiments.runtime.candidate_classifier_construction import (
    create_independent_candidate_training_state,
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


def build_candidate_epoch_training_oracle(*, class_count, optimizer_variant, monkeypatch):
    construction_arguments, legacy_client, legacy_initial_parameter_snapshot = (
        build_candidate_construction_oracle(
            class_count=class_count, optimizer_variant=optimizer_variant, monkeypatch=monkeypatch
        )
    )
    initial_rng_state = torch.get_rng_state().clone()
    legacy_candidate = create_legacy_candidate(
        legacy_client=legacy_client,
        legacy_initial_parameter_snapshot=legacy_initial_parameter_snapshot,
    )
    torch.set_rng_state(initial_rng_state)
    candidate_training_state = create_independent_candidate_training_state(**construction_arguments)
    return candidate_training_state, legacy_client, legacy_candidate


def assert_candidate_epoch_training_matches_legacy(*, candidate_training_state, legacy_candidate):
    for parameter_values, expected_parameter_values in zip(
        candidate_training_state.candidate_classifier.parameters(),
        legacy_candidate.parameters(),
        strict=True,
    ):
        assert torch.equal(parameter_values, expected_parameter_values)
        if expected_parameter_values.grad is None:
            assert parameter_values.grad is None
        else:
            assert torch.equal(parameter_values.grad, expected_parameter_values.grad)
    assert_nested_state_equal(
        candidate_training_state.candidate_shared_parameter_optimizer_state.parameter_optimizer.state_dict(),
        legacy_candidate.backbone.optimizer.state_dict(),
    )
    assert_nested_state_equal(
        candidate_training_state.candidate_concept_specific_parameter_optimizer_state.parameter_optimizer.state_dict(),
        legacy_candidate.head_optimizer.state_dict(),
    )


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize("optimizer_variant", ["standard", "amsgrad", "sgd"])
@pytest.mark.parametrize("epoch_count", [0, 1, 3])
@pytest.mark.parametrize("sample_count,maximum_batch_sample_count", [(7, 3), (11, 5), (4, 8)])
def test_candidate_dataset_epochs_matches_legacy(
    class_count,
    optimizer_variant,
    epoch_count,
    sample_count,
    maximum_batch_sample_count,
    monkeypatch,
):
    initial_rng_state = torch.get_rng_state().clone()
    try:
        candidate_training_state, legacy_client, legacy_candidate = (
            build_candidate_epoch_training_oracle(
                class_count=class_count,
                optimizer_variant=optimizer_variant,
                monkeypatch=monkeypatch,
            )
        )
        input_features = torch.arange(sample_count * 2, dtype=torch.float32).reshape(-1, 2) / 7
        observed_class_labels = (torch.arange(sample_count) % class_count).float().reshape(-1, 1)
        training_dataset = torch.utils.data.TensorDataset(input_features, observed_class_labels)
        input_tensor_values = tuple(
            training_tensor.clone() for training_tensor in training_dataset.tensors
        )
        monkeypatch.setattr(config, "CLIENT_BATCH_SIZE", maximum_batch_sample_count)
        initial_training_rng_state = torch.get_rng_state().clone()
        with patch.object(
            legacy_candidate, "update", wraps=legacy_candidate.update
        ) as record_legacy_candidate_dataset_epochs:
            BaseClient._update_new_model_epochs(
                legacy_client, legacy_candidate, training_dataset, epoch_count
            )
        expected_rng_state = torch.get_rng_state().clone()
        torch.set_rng_state(initial_training_rng_state)
        with (
            patch(
                "federated_learning_experiments.learning.training.candidate_epoch_training.perform_joint_model_parameter_update",
                wraps=perform_joint_model_parameter_update,
            ) as record_candidate_dataset_epochs,
            patch(
                "federated_learning_experiments.learning.training.candidate_epoch_training.DataLoader",
                wraps=torch.utils.data.DataLoader,
            ) as training_data_loader,
        ):
            epoch_training_result = _train_candidate_dataset_epochs(
                candidate_classifier=candidate_training_state.candidate_classifier,
                candidate_shared_parameter_optimizer_state=candidate_training_state.candidate_shared_parameter_optimizer_state,
                candidate_concept_specific_parameter_optimizer_state=candidate_training_state.candidate_concept_specific_parameter_optimizer_state,
                training_dataset=training_dataset,
                epoch_count=epoch_count,
                maximum_batch_sample_count=maximum_batch_sample_count,
            )
        training_data_loader.assert_called_once_with(
            training_dataset, batch_size=min(maximum_batch_sample_count, sample_count), shuffle=True
        )
        assert torch.equal(torch.get_rng_state(), expected_rng_state)
        assert len(record_candidate_dataset_epochs.call_args_list) == len(
            record_legacy_candidate_dataset_epochs.call_args_list
        )
        for training_batch_index in range(len(record_candidate_dataset_epochs.call_args_list)):
            assert (
                len(
                    record_candidate_dataset_epochs.call_args_list[training_batch_index].kwargs[
                        "participating_training_batches"
                    ]
                )
                == 1
            )
            participating_training_batch = record_candidate_dataset_epochs.call_args_list[
                training_batch_index
            ].kwargs["participating_training_batches"][0]
            recorded_training_datasets = record_legacy_candidate_dataset_epochs.call_args_list[
                training_batch_index
            ].args
            assert torch.equal(
                participating_training_batch.input_features, recorded_training_datasets[0]
            )
            assert torch.equal(
                participating_training_batch.observed_class_labels, recorded_training_datasets[1]
            )
        assert_candidate_epoch_training_matches_legacy(
            candidate_training_state=candidate_training_state, legacy_candidate=legacy_candidate
        )
        assert epoch_training_result == CandidateEpochTrainingResult(
            completed_epoch_count=epoch_count,
            candidate_trained_sample_count=legacy_client.compute_counters["training_examples"],
            candidate_parameter_update_step_count=legacy_client.compute_counters["optimizer_steps"],
            validation_evaluated_sample_count=0,
        )
        for training_tensor, parameter_values in zip(
            training_dataset.tensors, input_tensor_values, strict=True
        ):
            assert torch.equal(training_tensor, parameter_values)
    finally:
        torch.set_rng_state(initial_rng_state)


def test_candidate_epoch_training_result_contract():
    epoch_training_result = CandidateEpochTrainingResult(
        completed_epoch_count=1,
        candidate_trained_sample_count=3,
        candidate_parameter_update_step_count=1,
        validation_evaluated_sample_count=0,
    )
    assert all(
        configuration_parameter_name.kw_only
        for configuration_parameter_name in fields(epoch_training_result)
    )
    assert tuple(
        configuration_parameter_name.name
        for configuration_parameter_name in fields(epoch_training_result)
    ) == (
        "completed_epoch_count",
        "candidate_trained_sample_count",
        "candidate_parameter_update_step_count",
        "validation_evaluated_sample_count",
    )
    with pytest.raises(TypeError):
        CandidateEpochTrainingResult(1, 3, 1, 0)
    with pytest.raises(FrozenInstanceError):
        epoch_training_result.completed_epoch_count = 2


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize("optimizer_variant", ["standard", "amsgrad", "sgd"])
@pytest.mark.parametrize(
    "candidate_training_strategy",
    ["fixed_epoch_training", "validation_loss_early_stopping", "skip_training"],
)
@pytest.mark.parametrize("maximum_epoch_count", [0, 1, 4])
@pytest.mark.parametrize(
    "sample_count,validation_sample_fraction", [(1, 0.5), (3, 0.5), (5, 0.5), (11, 0.2)]
)
def test_candidate_epoch_training_matches_legacy(
    class_count,
    optimizer_variant,
    candidate_training_strategy,
    maximum_epoch_count,
    sample_count,
    validation_sample_fraction,
    monkeypatch,
    minimum_validation_loss_decrease=0.0001,
):
    initial_rng_state = torch.get_rng_state().clone()
    try:
        candidate_training_state, legacy_client, legacy_candidate = (
            build_candidate_epoch_training_oracle(
                class_count=class_count,
                optimizer_variant=optimizer_variant,
                monkeypatch=monkeypatch,
            )
        )
        input_features = torch.arange(sample_count * 2, dtype=torch.float32).reshape(-1, 2) / 7
        observed_class_labels = (torch.arange(sample_count) % class_count).float().reshape(-1, 1)
        candidate_epoch_training_settings = CandidateEpochTrainingSettings(
            candidate_training_strategy=candidate_training_strategy,
            maximum_epoch_count=maximum_epoch_count,
            maximum_batch_sample_count=3,
            validation_sample_fraction=validation_sample_fraction,
            consecutive_non_improving_epoch_limit=1,
            minimum_validation_loss_decrease=minimum_validation_loss_decrease,
        )
        monkeypatch.setattr(
            config,
            "NEW_MODEL_TRAINING",
            {
                "fixed_epoch_training": "fixed",
                "validation_loss_early_stopping": "early_stopping",
                "skip_training": "none",
            }[candidate_training_strategy],
        )
        monkeypatch.setattr(config, "NEW_MODEL_EPOCHS", maximum_epoch_count)
        monkeypatch.setattr(config, "CLIENT_BATCH_SIZE", 3)
        monkeypatch.setattr(config, "NEW_MODEL_VALIDATION_FRACTION", validation_sample_fraction)
        monkeypatch.setattr(config, "NEW_MODEL_EARLY_STOPPING_PATIENCE", 1)
        monkeypatch.setattr(
            config, "NEW_MODEL_EARLY_STOPPING_MIN_DELTA", minimum_validation_loss_decrease
        )
        input_tensor_values = (input_features.clone(), observed_class_labels.clone())
        candidate_parameter_values_and_gradients = snapshot_parameter_values_and_gradients(
            tuple(candidate_training_state.candidate_classifier.parameters())
        )
        legacy_validation_losses = []
        recorded_validation_losses = []

        def record_legacy_candidate_validation_losses(input_features, observed_class_labels):
            losses = type(legacy_candidate).per_sample_error(
                legacy_candidate, input_features, observed_class_labels
            )
            legacy_validation_losses.append(float(losses.mean().item()))
            return losses

        def record_candidate_validation_losses(
            *, classifier, input_features, observed_class_labels
        ):
            losses = evaluate_classifier_per_sample_bounded_losses(
                classifier=classifier,
                input_features=input_features,
                observed_class_labels=observed_class_labels,
            )
            recorded_validation_losses.append(float(losses.mean().item()))
            return losses

        initial_training_rng_state = torch.get_rng_state().clone()
        with (
            patch.object(
                legacy_client,
                "_update_new_model_epochs",
                wraps=legacy_client._update_new_model_epochs,
            ) as record_legacy_candidate_dataset_epochs,
            patch.object(
                legacy_candidate,
                "per_sample_error",
                side_effect=record_legacy_candidate_validation_losses,
            ),
        ):
            BaseClient._train_new_model(
                legacy_client, legacy_candidate, input_features, observed_class_labels
            )
        expected_rng_state = torch.get_rng_state().clone()
        recorded_epoch_counts = [
            record_legacy_candidate_dataset_epochs.call_args_list[training_batch_index].args[2]
            for training_batch_index in range(
                len(record_legacy_candidate_dataset_epochs.call_args_list)
            )
        ]
        recorded_training_datasets = [
            record_legacy_candidate_dataset_epochs.call_args_list[training_batch_index]
            .args[1]
            .tensors
            for training_batch_index in range(
                len(record_legacy_candidate_dataset_epochs.call_args_list)
            )
        ]
        expected_training_result = CandidateEpochTrainingResult(
            completed_epoch_count=sum(recorded_epoch_counts),
            candidate_trained_sample_count=legacy_client.compute_counters["training_examples"],
            candidate_parameter_update_step_count=legacy_client.compute_counters["optimizer_steps"],
            validation_evaluated_sample_count=legacy_client.compute_counters[
                "initialization_examples"
            ],
        )
        torch.set_rng_state(initial_training_rng_state)
        with (
            patch(
                "federated_learning_experiments.learning.training.candidate_epoch_training._train_candidate_dataset_epochs",
                wraps=_train_candidate_dataset_epochs,
            ) as record_candidate_dataset_epochs,
            patch(
                "federated_learning_experiments.learning.training.candidate_epoch_training.evaluate_classifier_per_sample_bounded_losses",
                side_effect=record_candidate_validation_losses,
            ),
        ):
            epoch_training_result = train_candidate_classifier_epochs(
                candidate_classifier=candidate_training_state.candidate_classifier,
                candidate_shared_parameter_optimizer_state=candidate_training_state.candidate_shared_parameter_optimizer_state,
                candidate_concept_specific_parameter_optimizer_state=candidate_training_state.candidate_concept_specific_parameter_optimizer_state,
                input_features=input_features,
                observed_class_labels=observed_class_labels,
                candidate_epoch_training_settings=candidate_epoch_training_settings,
            )
        assert epoch_training_result == expected_training_result
        assert recorded_validation_losses == legacy_validation_losses
        assert [
            record_candidate_dataset_epochs.call_args_list[training_batch_index].kwargs[
                "epoch_count"
            ]
            for training_batch_index in range(len(record_candidate_dataset_epochs.call_args_list))
        ] == recorded_epoch_counts
        for training_batch_index in range(len(record_candidate_dataset_epochs.call_args_list)):
            training_arguments = record_candidate_dataset_epochs.call_args_list[
                training_batch_index
            ].kwargs
            for training_tensor, parameter_values in zip(
                training_arguments["training_dataset"].tensors,
                recorded_training_datasets[training_batch_index],
                strict=True,
            ):
                assert torch.equal(training_tensor, parameter_values)
        assert torch.equal(torch.get_rng_state(), expected_rng_state)
        assert_candidate_epoch_training_matches_legacy(
            candidate_training_state=candidate_training_state, legacy_candidate=legacy_candidate
        )
        assert torch.equal(input_features, input_tensor_values[0])
        assert torch.equal(observed_class_labels, input_tensor_values[1])
        if candidate_training_strategy == "skip_training" or maximum_epoch_count == 0:
            assert_parameter_values_and_gradients_unchanged(
                candidate_parameter_values_and_gradients
            )
        if candidate_training_strategy == "skip_training" or (
            maximum_epoch_count == 0
            and (candidate_training_strategy == "fixed_epoch_training" or sample_count == 1)
        ):
            assert torch.equal(torch.get_rng_state(), initial_training_rng_state)
        else:
            assert not torch.equal(torch.get_rng_state(), initial_training_rng_state)
        if (
            candidate_training_strategy == "validation_loss_early_stopping"
            and sample_count > 1
            and maximum_epoch_count == 4
            and minimum_validation_loss_decrease == 1e6
        ):
            assert epoch_training_result.completed_epoch_count == 2
    finally:
        torch.set_rng_state(initial_rng_state)


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize("optimizer_variant", ["standard", "amsgrad", "sgd"])
def test_candidate_epoch_training_restores_only_best_parameters(
    class_count, optimizer_variant, monkeypatch
):
    test_candidate_epoch_training_matches_legacy(
        class_count,
        optimizer_variant,
        "validation_loss_early_stopping",
        4,
        11,
        0.2,
        monkeypatch,
        minimum_validation_loss_decrease=1e6,
    )
    initial_rng_state = torch.get_rng_state().clone()
    try:
        candidate_training_state, legacy_client, legacy_candidate = (
            build_candidate_epoch_training_oracle(
                class_count=class_count,
                optimizer_variant=optimizer_variant,
                monkeypatch=monkeypatch,
            )
        )
        input_features = torch.ones(9, 2)
        observed_class_labels = torch.zeros(9, 1)
        monkeypatch.setattr(config, "NEW_MODEL_TRAINING", "early_stopping")
        monkeypatch.setattr(config, "NEW_MODEL_EPOCHS", 2)
        monkeypatch.setattr(config, "CLIENT_BATCH_SIZE", 3)
        monkeypatch.setattr(config, "NEW_MODEL_VALIDATION_FRACTION", 1 / 3)
        monkeypatch.setattr(config, "NEW_MODEL_EARLY_STOPPING_PATIENCE", 1)
        monkeypatch.setattr(config, "NEW_MODEL_EARLY_STOPPING_MIN_DELTA", 0)
        legacy_validation_losses = []

        def record_legacy_candidate_validation_losses(input_features, observed_class_labels):
            losses = type(legacy_candidate).per_sample_error(
                legacy_candidate, input_features, observed_class_labels
            )
            legacy_validation_losses.append(float(losses.mean().item()))
            return losses

        with patch.object(
            legacy_candidate,
            "per_sample_error",
            side_effect=record_legacy_candidate_validation_losses,
        ):
            BaseClient._train_new_model(
                legacy_client, legacy_candidate, input_features, observed_class_labels
            )
        assert len(legacy_validation_losses) == 2
        validation_loss_decrease_at_boundary = (
            legacy_validation_losses[0] - legacy_validation_losses[1]
        )
        assert validation_loss_decrease_at_boundary > 0
        assert (
            legacy_validation_losses[1]
            == legacy_validation_losses[0] - validation_loss_decrease_at_boundary
        )
        torch.set_rng_state(initial_rng_state)
        candidate_training_state, legacy_client, legacy_candidate = (
            build_candidate_epoch_training_oracle(
                class_count=class_count,
                optimizer_variant=optimizer_variant,
                monkeypatch=monkeypatch,
            )
        )
        monkeypatch.setattr(config, "NEW_MODEL_EPOCHS", 4)
        monkeypatch.setattr(
            config, "NEW_MODEL_EARLY_STOPPING_MIN_DELTA", validation_loss_decrease_at_boundary
        )
        candidate_epoch_training_settings = CandidateEpochTrainingSettings(
            candidate_training_strategy="validation_loss_early_stopping",
            maximum_epoch_count=4,
            maximum_batch_sample_count=3,
            validation_sample_fraction=1 / 3,
            consecutive_non_improving_epoch_limit=1,
            minimum_validation_loss_decrease=validation_loss_decrease_at_boundary,
        )
        initial_training_rng_state = torch.get_rng_state().clone()
        BaseClient._train_new_model(
            legacy_client, legacy_candidate, input_features, observed_class_labels
        )
        expected_rng_state = torch.get_rng_state().clone()
        torch.set_rng_state(initial_training_rng_state)
        epoch_training_result = train_candidate_classifier_epochs(
            candidate_classifier=candidate_training_state.candidate_classifier,
            candidate_shared_parameter_optimizer_state=candidate_training_state.candidate_shared_parameter_optimizer_state,
            candidate_concept_specific_parameter_optimizer_state=candidate_training_state.candidate_concept_specific_parameter_optimizer_state,
            input_features=input_features,
            observed_class_labels=observed_class_labels,
            candidate_epoch_training_settings=candidate_epoch_training_settings,
        )
        assert epoch_training_result.completed_epoch_count == 2
        assert (
            epoch_training_result.candidate_trained_sample_count
            == legacy_client.compute_counters["training_examples"]
        )
        assert (
            epoch_training_result.candidate_parameter_update_step_count
            == legacy_client.compute_counters["optimizer_steps"]
        )
        assert (
            epoch_training_result.validation_evaluated_sample_count
            == legacy_client.compute_counters["initialization_examples"]
        )
        assert torch.equal(torch.get_rng_state(), expected_rng_state)
        assert_candidate_epoch_training_matches_legacy(
            candidate_training_state=candidate_training_state, legacy_candidate=legacy_candidate
        )
    finally:
        torch.set_rng_state(initial_rng_state)


@pytest.mark.parametrize(
    "invalid_case",
    [
        "valid",
        "settings_type",
        "settings_subclass",
        "settings_value",
        "classifier_type",
        "classifier_subclass",
        "classifier_dtype",
        "classifier_nonfinite",
        "empty_shared",
        "shared_owner_type",
        "concept_owner_type",
        "shared_owner_subclass",
        "concept_owner_subclass",
        "shared_optimizer_subclass",
        "concept_optimizer_subclass",
        "shared_binding",
        "concept_binding",
        "shared_order",
        "concept_order",
        "same_optimizer",
        "frozen_parameter",
        "features_type",
        "labels_type",
        "features_dtype",
        "labels_dtype",
        "features_shape",
        "labels_shape",
        "labels_count",
        "features_device",
        "labels_device",
        "features_sparse",
        "labels_sparse",
        "features_nested",
        "labels_nested",
        "empty",
        "nan",
        "labels_nonfinite",
        "label_fraction",
        "label_negative",
        "label_upper",
        "grad_disabled",
        "skip_grad_disabled",
        "zero_grad_disabled",
    ],
)
def test_candidate_epoch_training_rejects_before_mutation(
    invalid_case, valid_settings_arguments, monkeypatch
):
    initial_rng_state = torch.get_rng_state().clone()
    try:
        candidate_training_state, legacy_client, legacy_candidate = (
            build_candidate_epoch_training_oracle(
                class_count=4, optimizer_variant="standard", monkeypatch=monkeypatch
            )
        )
        input_features = torch.arange(14, dtype=torch.float32).reshape(-1, 2) / 7
        observed_class_labels = (torch.arange(7) % 4).float().reshape(-1, 1)
        _train_candidate_dataset_epochs(
            candidate_classifier=candidate_training_state.candidate_classifier,
            candidate_shared_parameter_optimizer_state=candidate_training_state.candidate_shared_parameter_optimizer_state,
            candidate_concept_specific_parameter_optimizer_state=candidate_training_state.candidate_concept_specific_parameter_optimizer_state,
            training_dataset=torch.utils.data.TensorDataset(input_features, observed_class_labels),
            epoch_count=1,
            maximum_batch_sample_count=3,
        )
        training_arguments = dict(
            candidate_classifier=candidate_training_state.candidate_classifier,
            candidate_shared_parameter_optimizer_state=candidate_training_state.candidate_shared_parameter_optimizer_state,
            candidate_concept_specific_parameter_optimizer_state=candidate_training_state.candidate_concept_specific_parameter_optimizer_state,
            input_features=input_features,
            observed_class_labels=observed_class_labels,
            candidate_epoch_training_settings=CandidateEpochTrainingSettings(
                **valid_settings_arguments
            ),
        )
        if invalid_case == "settings_type":
            training_arguments["candidate_epoch_training_settings"] = object()
        elif invalid_case == "settings_value":
            object.__setattr__(
                training_arguments["candidate_epoch_training_settings"], "maximum_epoch_count", -1
            )
        elif invalid_case == "settings_subclass":
            training_arguments["candidate_epoch_training_settings"] = type(
                "SettingsSubclass", (CandidateEpochTrainingSettings,), {}
            )(**valid_settings_arguments)
        elif invalid_case == "classifier_type":
            training_arguments["candidate_classifier"] = object()
        elif invalid_case == "classifier_subclass":
            candidate_training_state.candidate_classifier.__class__ = type(
                "ClassifierSubclass", (type(candidate_training_state.candidate_classifier),), {}
            )
        elif invalid_case == "classifier_dtype":
            candidate_training_state.candidate_classifier.double()
        elif invalid_case == "classifier_nonfinite":
            with torch.no_grad():
                next(candidate_training_state.candidate_classifier.parameters()).fill_(float("nan"))
        elif invalid_case == "empty_shared":
            candidate_training_state.candidate_classifier.feature_extractor = type(
                candidate_training_state.candidate_classifier.feature_extractor
            )(input_feature_count=2, hidden_layer_widths=())
        elif invalid_case in ("shared_owner_type", "concept_owner_type"):
            training_arguments[
                "candidate_shared_parameter_optimizer_state"
                if invalid_case == "shared_owner_type"
                else "candidate_concept_specific_parameter_optimizer_state"
            ] = object()
        elif invalid_case in (
            "shared_owner_subclass",
            "concept_owner_subclass",
            "shared_optimizer_subclass",
            "concept_optimizer_subclass",
        ):
            parameter_optimizer_state = (
                candidate_training_state.candidate_shared_parameter_optimizer_state
                if invalid_case.startswith("shared")
                else candidate_training_state.candidate_concept_specific_parameter_optimizer_state
            )
            if "owner" in invalid_case:
                parameter_optimizer_state.__class__ = type(
                    "OptimizerStateSubclass", (type(parameter_optimizer_state),), {}
                )
            else:
                parameter_optimizer_state.parameter_optimizer.__class__ = type(
                    "OptimizerSubclass", (type(parameter_optimizer_state.parameter_optimizer),), {}
                )
        elif invalid_case in (
            "shared_binding",
            "shared_order",
            "concept_binding",
            "concept_order",
            "same_optimizer",
        ):
            parameter_optimizer_state = (
                candidate_training_state.candidate_shared_parameter_optimizer_state
                if invalid_case.startswith("shared")
                else candidate_training_state.candidate_concept_specific_parameter_optimizer_state
            )
            optimizer_parameters = parameter_optimizer_state.parameter_optimizer.param_groups[0][
                "params"
            ]
            if invalid_case.endswith("order"):
                optimizer_parameters.reverse()
            elif invalid_case == "same_optimizer":
                training_arguments["candidate_concept_specific_parameter_optimizer_state"] = (
                    candidate_training_state.candidate_shared_parameter_optimizer_state
                )
            else:
                classifier_parameter = optimizer_parameters[0]
                optimizer_parameters[0] = torch.nn.Parameter(
                    optimizer_parameters[0].detach().clone()
                )
                parameter_optimizer_state.parameter_optimizer.state[optimizer_parameters[0]] = (
                    parameter_optimizer_state.parameter_optimizer.state.pop(classifier_parameter)
                )
        elif invalid_case == "frozen_parameter":
            next(candidate_training_state.candidate_classifier.parameters()).requires_grad_(False)
        elif invalid_case in ("features_type", "labels_type"):
            training_arguments[
                "input_features" if invalid_case == "features_type" else "observed_class_labels"
            ] = []
        elif invalid_case in ("features_dtype", "labels_dtype"):
            training_arguments[
                "input_features" if invalid_case == "features_dtype" else "observed_class_labels"
            ] = (
                input_features if invalid_case == "features_dtype" else observed_class_labels
            ).double()
        elif invalid_case == "features_shape":
            training_arguments["input_features"] = input_features[:, :1]
        elif invalid_case == "labels_shape":
            training_arguments["observed_class_labels"] = observed_class_labels.flatten()
        elif invalid_case == "labels_count":
            training_arguments["observed_class_labels"] = observed_class_labels[:1]
        elif invalid_case.endswith("device"):
            training_arguments[
                "input_features" if invalid_case.startswith("features") else "observed_class_labels"
            ] = (
                input_features if invalid_case.startswith("features") else observed_class_labels
            ).to("meta")
        elif invalid_case.endswith("sparse"):
            training_arguments[
                "input_features" if invalid_case.startswith("features") else "observed_class_labels"
            ] = (
                input_features if invalid_case.startswith("features") else observed_class_labels
            ).to_sparse()
        elif invalid_case.endswith("nested"):
            training_arguments[
                "input_features" if invalid_case.startswith("features") else "observed_class_labels"
            ] = torch.nested.nested_tensor(
                [input_features if invalid_case.startswith("features") else observed_class_labels]
            )
        elif invalid_case == "empty":
            training_arguments["input_features"] = input_features[:0]
            training_arguments["observed_class_labels"] = observed_class_labels[:0]
        elif invalid_case == "nan":
            input_features[0, 0] = float("nan")
        elif invalid_case == "labels_nonfinite":
            observed_class_labels[0, 0] = float("inf")
        elif invalid_case in ("label_fraction", "label_negative", "label_upper"):
            observed_class_labels[0, 0] = {
                "label_fraction": 0.5,
                "label_negative": -1,
                "label_upper": 4,
            }[invalid_case]
        elif invalid_case in ("skip_grad_disabled", "zero_grad_disabled"):
            valid_settings_arguments["candidate_training_strategy"] = (
                "skip_training" if invalid_case == "skip_grad_disabled" else "fixed_epoch_training"
            )
            valid_settings_arguments["maximum_epoch_count"] = (
                0 if invalid_case == "zero_grad_disabled" else 1
            )
            training_arguments["candidate_epoch_training_settings"] = (
                CandidateEpochTrainingSettings(**valid_settings_arguments)
            )
        candidate_parameter_values_and_gradients = snapshot_parameter_values_and_gradients(
            tuple(candidate_training_state.candidate_classifier.parameters())
        )
        input_tensor_values = (input_features.clone(), observed_class_labels.clone())
        shared_optimizer_state_snapshot = deepcopy(
            candidate_training_state.candidate_shared_parameter_optimizer_state.parameter_optimizer.state_dict()
        )
        concept_specific_optimizer_state_snapshot = deepcopy(
            candidate_training_state.candidate_concept_specific_parameter_optimizer_state.parameter_optimizer.state_dict()
        )
        expected_rng_state = torch.get_rng_state().clone()
        if invalid_case in ("valid", "skip_grad_disabled", "zero_grad_disabled"):
            with torch.set_grad_enabled(invalid_case == "valid"):
                if invalid_case == "valid":
                    _validate_candidate_epoch_training_inputs(**training_arguments)
                else:
                    train_candidate_classifier_epochs(**training_arguments)
        else:
            with (
                torch.set_grad_enabled(invalid_case != "grad_disabled"),
                pytest.raises(
                    TypeError
                    if invalid_case
                    in (
                        "settings_type",
                        "classifier_type",
                        "shared_owner_type",
                        "concept_owner_type",
                        "features_type",
                        "labels_type",
                    )
                    or invalid_case.endswith("subclass")
                    else ValueError
                ),
            ):
                train_candidate_classifier_epochs(**training_arguments)
        assert torch.equal(torch.get_rng_state(), expected_rng_state)
        if invalid_case == "classifier_nonfinite":
            for (
                parameter_values,
                expected_parameter_values,
                previous_gradients,
            ) in candidate_parameter_values_and_gradients:
                torch.testing.assert_close(
                    parameter_values, expected_parameter_values, rtol=0, atol=0, equal_nan=True
                )
                assert parameter_values.grad is previous_gradients[0]
                if previous_gradients[1] is not None:
                    assert torch.equal(parameter_values.grad, previous_gradients[1])
        else:
            assert_parameter_values_and_gradients_unchanged(
                candidate_parameter_values_and_gradients
            )
        torch.testing.assert_close(
            input_features, input_tensor_values[0], rtol=0, atol=0, equal_nan=True
        )
        assert torch.equal(observed_class_labels, input_tensor_values[1])
        assert_nested_state_equal(
            candidate_training_state.candidate_shared_parameter_optimizer_state.parameter_optimizer.state_dict(),
            shared_optimizer_state_snapshot,
        )
        assert_nested_state_equal(
            candidate_training_state.candidate_concept_specific_parameter_optimizer_state.parameter_optimizer.state_dict(),
            concept_specific_optimizer_state_snapshot,
        )
    finally:
        torch.set_rng_state(initial_rng_state)
