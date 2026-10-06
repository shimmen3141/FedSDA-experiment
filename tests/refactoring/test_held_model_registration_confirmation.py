"""正式登録確認による複数ownerの更新を実旧処理と対照する。"""

import random
from copy import deepcopy
from dataclasses import replace
from random import Random

import numpy as np
import pytest
import torch
from test_held_model_training_state_registry import (
    build_registry_classifier_and_owner,
    register_training_state,
)
from test_joint_model_parameter_update import (
    assert_joint_update_states_equal,
    assert_nested_state_equal,
    build_joint_update_oracle_pair,
    run_legacy_joint_update,
)
from test_loss_statistics_model_id_reassignment import assert_store_statistics_match_legacy
from test_model_training_and_assignment_counts import (
    assert_model_counts_match_legacy,
    build_model_counts_oracle,
    record_concept_counts_in_both_implementations,
    record_training_counts_in_both_implementations,
)
from test_shared_feature_extractor_attachment import (
    assert_parameter_values_and_gradients_unchanged,
    snapshot_parameter_values_and_gradients,
)

from federated_drift_experiment.clients.base import BaseClient
from federated_learning_experiments.evaluation.model_evaluation_sample_records import (
    ObservedEvaluationSample,
)
from federated_learning_experiments.evaluation.model_evaluation_sample_store import (
    ModelEvaluationSampleStore,
)
from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatisticsStore,
)
from federated_learning_experiments.learning.models.classifier_parameter_snapshot import (
    snapshot_classifier_parameters,
)
from federated_learning_experiments.learning.training.current_training_model_assignment import (
    CurrentTrainingModelAssignment,
)
from federated_learning_experiments.learning.training.held_model_training_state_registry import (
    HeldModelTrainingStateRegistry,
)
from federated_learning_experiments.learning.training.joint_model_parameter_update import (
    perform_joint_model_parameter_update,
)
from federated_learning_experiments.learning.training.local_training_settings import (
    LocalTrainingSettings,
)
from federated_learning_experiments.learning.training.model_training_sample_records import (
    ObservedTrainingSample,
)
from federated_learning_experiments.learning.training.model_training_sample_store import (
    ModelTrainingSampleStore,
)
from federated_learning_experiments.learning.training.parameter_optimizer_settings import (
    AdamParameterOptimizerSettings,
    SgdParameterOptimizerSettings,
)
from federated_learning_experiments.learning.training.parameter_optimizer_state import (
    ParameterOptimizerState,
)
from federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload import (
    PendingModelUploadState,
)
from federated_learning_experiments.runtime.held_model_registration_confirmation import (
    confirm_held_model_registration,
)


class IntSubclass(int):
    pass


def build_registration_confirmation_oracle(
    *, initial_model_ids=(4, -7, 9), auxiliary_source_present=True, receiving_model_id=12
):
    counts_store, legacy_client = build_model_counts_oracle()
    legacy_client.current_model_id = -7
    registry = HeldModelTrainingStateRegistry()
    loss_statistics_store = ModelAndClassLossStatisticsStore()
    training_sample_store = ModelTrainingSampleStore()
    evaluation_sample_store = ModelEvaluationSampleStore(
        maximum_stored_sample_count_per_model=3, added_batch_sample_count=1
    )
    for model_id in initial_model_ids:
        classifier, optimizer_owner = build_registry_classifier_and_owner()
        register_training_state(registry, model_id, classifier, optimizer_owner)
        legacy_client.models[model_id] = classifier
        if model_id == -7 and not auxiliary_source_present:
            continue
        loss_statistics_store.record_assigned_loss(
            model_id=model_id, observed_loss=0.25, observed_class_id=0
        )
        legacy_client.model_stats[model_id] = {
            "n": 1,
            "mean": 0.25,
            "M2": 0.0,
            "class_stats": {0: {"n": 1, "mean": 0.25, "M2": 0.0}},
        }
        training_samples = (
            ObservedTrainingSample(
                input_features=torch.tensor([[float(model_id), 1.0]]),
                observed_class_labels=torch.tensor([0]),
            ),
        )
        training_sample_store.append_model_training_samples(
            model_id=model_id, training_samples=training_samples
        )
        legacy_client.train_data_store[model_id] = list(training_samples)
        evaluation_samples = (
            ObservedEvaluationSample(
                input_features=training_samples[0].input_features,
                observed_class_labels=training_samples[0].observed_class_labels,
            ),
        )
        evaluation_sample_store.sample_and_append_model_evaluation_samples(
            model_id=99 if model_id < 0 else model_id,
            evaluation_samples=evaluation_samples,
            python_random_generator=Random(1),
        )
        if model_id < 0:
            evaluation_sample_store.reassign_model_evaluation_samples_id(
                original_model_id=99, reassigned_model_id=model_id
            )
        legacy_client.stored_data[model_id] = list(evaluation_samples)
        record_training_counts_in_both_implementations(
            counts_store=counts_store,
            legacy_client=legacy_client,
            model_id=model_id,
            trained_sample_count=3,
            parameter_update_step_count=2,
        )
        record_concept_counts_in_both_implementations(
            counts_store=counts_store,
            legacy_client=legacy_client,
            model_id=model_id,
            observed_concept_id=0,
        )
    classifier = legacy_client.models.get(-7, next(iter(legacy_client.models.values())))
    pending_upload_state = PendingModelUploadState()
    pending_upload_state.queue_model_upload(
        model_id=-7,
        parameter_snapshot=snapshot_classifier_parameters(classifier=classifier),
        upload_delay_round_count=2,
    )
    legacy_client.pending_model_params = (
        pending_upload_state.get_pending_model_upload().parameter_snapshot
    )
    legacy_client.pending_model_stats = legacy_client.model_stats.get(-7)
    legacy_client.pending_model_ready = False
    confirmation_arguments = dict(
        registered_global_model_id=receiving_model_id,
        held_model_training_state_registry=registry,
        loss_statistics_store=loss_statistics_store,
        training_sample_store=training_sample_store,
        evaluation_sample_store=evaluation_sample_store,
        model_training_and_assignment_counts_store=counts_store,
        current_training_model_assignment=CurrentTrainingModelAssignment(initial_model_id=-7),
        pending_model_upload_state=pending_upload_state,
    )
    return confirmation_arguments, legacy_client


def snapshot_confirmation_owners(confirmation_arguments):
    return (
        confirmation_arguments[
            "held_model_training_state_registry"
        ].snapshot_ordered_held_model_training_states(),
        confirmation_arguments["loss_statistics_store"].get_state_snapshot(),
        confirmation_arguments["training_sample_store"].snapshot_ordered_model_training_samples(),
        confirmation_arguments[
            "evaluation_sample_store"
        ].snapshot_ordered_model_evaluation_samples(),
        confirmation_arguments[
            "model_training_and_assignment_counts_store"
        ].snapshot_model_training_and_assignment_counts(),
        confirmation_arguments["current_training_model_assignment"].current_training_model_id,
        confirmation_arguments["pending_model_upload_state"].get_pending_model_upload(),
        confirmation_arguments["pending_model_upload_state"].remaining_upload_delay_round_count,
    )


def assert_confirmation_matches_legacy(
    confirmation_arguments, legacy_client, *, compare_classifier_identity=True
):
    registry = confirmation_arguments["held_model_training_state_registry"]
    assert tuple(
        state.model_id for state in registry.snapshot_ordered_held_model_training_states()
    ) == tuple(legacy_client.models)
    if compare_classifier_identity:
        for state in registry.snapshot_ordered_held_model_training_states():
            assert state.classifier is legacy_client.models[state.model_id]
    assert_store_statistics_match_legacy(
        loss_statistics_store=confirmation_arguments["loss_statistics_store"],
        legacy_client=legacy_client,
    )
    for operation_name, original_operation, arguments in (
        (
            "training_sample_store",
            "snapshot_ordered_model_training_samples",
            legacy_client.train_data_store,
        ),
        (
            "evaluation_sample_store",
            "snapshot_ordered_model_evaluation_samples",
            legacy_client.stored_data,
        ),
    ):
        previous_snapshot = getattr(confirmation_arguments[operation_name], original_operation)()
        assert tuple(state.model_id for state in previous_snapshot) == tuple(arguments)
        for state in previous_snapshot:
            training_samples = getattr(
                state, "training_samples", getattr(state, "evaluation_samples", ())
            )
            assert len(training_samples) == len(arguments[state.model_id])
            assert all(
                sample is expected
                for sample, expected in zip(training_samples, arguments[state.model_id])
            )
    assert_model_counts_match_legacy(
        counts_store=confirmation_arguments["model_training_and_assignment_counts_store"],
        legacy_client=legacy_client,
    )
    assert (
        confirmation_arguments["current_training_model_assignment"].current_training_model_id
        == legacy_client.current_model_id
    )
    assert confirmation_arguments["pending_model_upload_state"].get_pending_model_upload() is None
    assert (
        confirmation_arguments["pending_model_upload_state"].remaining_upload_delay_round_count == 0
    )
    assert legacy_client.pending_model_params is None
    assert legacy_client.pending_model_stats is None
    assert legacy_client.pending_model_ready
    legacy_client._record_adaptation_event.assert_not_called()


@pytest.mark.parametrize("initial_model_ids", [(-7,), (4, -7, 9), (-7, 4, 9), (4, 9, -7)])
@pytest.mark.parametrize("receiving_model_id", [0, 4, 12, 10**40])
@pytest.mark.parametrize("auxiliary_source_present", [True, False])
def test_confirmation_matches_old_held_model_branch(
    initial_model_ids, receiving_model_id, auxiliary_source_present
):
    with torch.random.fork_rng(devices=[]):
        confirmation_arguments, legacy_client = build_registration_confirmation_oracle(
            initial_model_ids=initial_model_ids,
            receiving_model_id=receiving_model_id,
            auxiliary_source_present=auxiliary_source_present,
        )
        previous_snapshot = snapshot_confirmation_owners(confirmation_arguments)
        parameter_snapshots = snapshot_parameter_values_and_gradients(
            parameter
            for state in previous_snapshot[0]
            for parameter in state.classifier.parameters()
        )
        assignment_change = confirm_held_model_registration(**confirmation_arguments)
        BaseClient.confirm_model_registration(legacy_client, receiving_model_id)
        assert assignment_change.previous_model_id == -7
        assert assignment_change.current_model_id == receiving_model_id
        assert_confirmation_matches_legacy(confirmation_arguments, legacy_client)
        assert_parameter_values_and_gradients_unchanged(parameter_snapshots)
        assert tuple(state.model_id for state in previous_snapshot[0]) == initial_model_ids
        assert previous_snapshot[6].model_id == -7


@pytest.mark.parametrize("original_model_id", [0, 4, 999])
def test_nonnegative_current_id_only_clears_pending(original_model_id):
    with torch.random.fork_rng(devices=[]):
        confirmation_arguments, legacy_client = build_registration_confirmation_oracle()
        confirmation_arguments["current_training_model_assignment"].assign_model_for_training(
            model_id=original_model_id
        )
        legacy_client.current_model_id = original_model_id
        previous_snapshot = snapshot_confirmation_owners(confirmation_arguments)
        assert confirm_held_model_registration(**confirmation_arguments) is None
        BaseClient.confirm_model_registration(legacy_client, 12)
        assert snapshot_confirmation_owners(confirmation_arguments)[:6] == previous_snapshot[:6]
        assert_confirmation_matches_legacy(confirmation_arguments, legacy_client)


@pytest.mark.parametrize(
    "invalid_global_model_id", [-7, -1, None, True, False, 1.0, "4", IntSubclass(4), np.int64(4)]
)
def test_invalid_formal_id_preserves_all_owners(invalid_global_model_id):
    with torch.random.fork_rng(devices=[]):
        confirmation_arguments, _ = build_registration_confirmation_oracle()
        previous_snapshot = snapshot_confirmation_owners(confirmation_arguments)
        confirmation_arguments["registered_global_model_id"] = invalid_global_model_id
        with pytest.raises(ValueError if type(invalid_global_model_id) is int else TypeError):
            confirm_held_model_registration(**confirmation_arguments)
        assert snapshot_confirmation_owners(confirmation_arguments) == previous_snapshot


@pytest.mark.parametrize("original_model_id", [-7, 0])
@pytest.mark.parametrize(
    "invalid_owner_name",
    [
        "held_model_training_state_registry",
        "loss_statistics_store",
        "training_sample_store",
        "evaluation_sample_store",
        "model_training_and_assignment_counts_store",
        "current_training_model_assignment",
        "pending_model_upload_state",
    ],
)
@pytest.mark.parametrize("invalid_owner", [None, object()])
def test_invalid_owner_is_rejected_before_any_mutation(
    original_model_id, invalid_owner_name, invalid_owner
):
    with torch.random.fork_rng(devices=[]):
        confirmation_arguments, _ = build_registration_confirmation_oracle()
        confirmation_arguments["current_training_model_assignment"].assign_model_for_training(
            model_id=original_model_id
        )
        previous_snapshot = snapshot_confirmation_owners(confirmation_arguments)
        arguments = dict(confirmation_arguments)
        arguments[invalid_owner_name] = invalid_owner
        with pytest.raises(TypeError):
            confirm_held_model_registration(**arguments)
        assert snapshot_confirmation_owners(confirmation_arguments) == previous_snapshot


def test_missing_temporary_model_is_rejected_before_updates():
    with torch.random.fork_rng(devices=[]):
        confirmation_arguments, _ = build_registration_confirmation_oracle(initial_model_ids=(4, 9))
        previous_snapshot = snapshot_confirmation_owners(confirmation_arguments)
        with pytest.raises(KeyError) as arguments:
            confirm_held_model_registration(**confirmation_arguments)
        assert arguments.value.args == (-7,)
        assert snapshot_confirmation_owners(confirmation_arguments) == previous_snapshot


@pytest.mark.parametrize("original_model_id", [-7, 0])
def test_existing_owner_api_call_order_is_explicit(original_model_id, monkeypatch):
    with torch.random.fork_rng(devices=[]):
        confirmation_arguments, _ = build_registration_confirmation_oracle()
        confirmation_arguments["current_training_model_assignment"].assign_model_for_training(
            model_id=original_model_id
        )
        actual_call_order = []
        for owner_name, operation_name in (
            ("held_model_training_state_registry", "get_held_model_training_state"),
            ("held_model_training_state_registry", "reassign_held_model_training_state_id"),
            ("loss_statistics_store", "reassign_model_loss_statistics_id"),
            ("training_sample_store", "reassign_model_training_samples_id"),
            ("evaluation_sample_store", "reassign_model_evaluation_samples_id"),
            (
                "model_training_and_assignment_counts_store",
                "transfer_model_training_and_assignment_counts",
            ),
            ("current_training_model_assignment", "assign_model_for_training"),
            ("pending_model_upload_state", "clear_pending_model_upload"),
        ):
            state_owner = confirmation_arguments[owner_name]
            original_operation = getattr(type(state_owner), operation_name)
            monkeypatch.setattr(
                type(state_owner),
                operation_name,
                lambda *arguments, operation_name=operation_name, original_operation=original_operation, **keyword_arguments: (
                    (
                        actual_call_order.append(operation_name),
                        original_operation(*arguments, **keyword_arguments),
                    )[1]
                ),
            )
        confirm_held_model_registration(**confirmation_arguments)
        expected_call_order = (
            [
                "get_held_model_training_state",
                "reassign_held_model_training_state_id",
                "reassign_model_loss_statistics_id",
                "reassign_model_training_samples_id",
                "reassign_model_evaluation_samples_id",
                "transfer_model_training_and_assignment_counts",
                "assign_model_for_training",
                "clear_pending_model_upload",
            ]
            if original_model_id < 0
            else ["clear_pending_model_upload"]
        )
        assert actual_call_order == expected_call_order


@pytest.mark.parametrize(
    "invalid_owner_name",
    [
        "held_model_training_state_registry",
        "loss_statistics_store",
        "training_sample_store",
        "evaluation_sample_store",
        "model_training_and_assignment_counts_store",
        "current_training_model_assignment",
        "pending_model_upload_state",
    ],
)
def test_owner_subclasses_are_rejected_without_mutation(invalid_owner_name):
    with torch.random.fork_rng(devices=[]):
        confirmation_arguments, _ = build_registration_confirmation_oracle()
        previous_snapshot = snapshot_confirmation_owners(confirmation_arguments)
        OwnerSubclass = type(
            "OwnerSubclass", (type(confirmation_arguments[invalid_owner_name]),), {}
        )
        arguments = dict(confirmation_arguments)
        arguments[invalid_owner_name] = object.__new__(OwnerSubclass)
        with pytest.raises(TypeError):
            confirm_held_model_registration(**arguments)
        assert snapshot_confirmation_owners(confirmation_arguments) == previous_snapshot


@pytest.mark.parametrize("operation_name", ["ready", "empty"])
def test_confirmation_clears_ready_or_empty_upload_state(operation_name):
    with torch.random.fork_rng(devices=[]):
        confirmation_arguments, legacy_client = build_registration_confirmation_oracle()
        pending_upload_state = confirmation_arguments["pending_model_upload_state"]
        if operation_name == "ready":
            pending_upload_state.advance_upload_readiness_at_round_boundary()
            pending_upload_state.advance_upload_readiness_at_round_boundary()
            assert pending_upload_state.has_ready_model_upload()
        else:
            pending_upload_state.clear_pending_model_upload()
            legacy_client.pending_model_params = None
            legacy_client.pending_model_stats = None
        legacy_client.pending_model_ready = True
        confirm_held_model_registration(**confirmation_arguments)
        BaseClient.confirm_model_registration(legacy_client, 12)
        assert_confirmation_matches_legacy(confirmation_arguments, legacy_client)


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize("optimizer_variant", ["standard", "amsgrad", "sgd"])
@pytest.mark.parametrize("update_shared_features", [True, False])
def test_confirmed_registration_continues_actual_joint_training(
    class_count, optimizer_variant, update_shared_features, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        training_batches, shared_optimizer, legacy_client = build_joint_update_oracle_pair(
            class_count=class_count,
            batch_sample_counts=(3, 5),
            optimizer_variant=optimizer_variant,
            monkeypatch=monkeypatch,
        )
        confirmation_arguments, legacy_registration_client = build_registration_confirmation_oracle(
            initial_model_ids=(4, -7)
        )
        registry = confirmation_arguments["held_model_training_state_registry"]
        optimizer_settings = (
            SgdParameterOptimizerSettings(learning_rate=0.01)
            if optimizer_variant == "sgd"
            else AdamParameterOptimizerSettings(
                learning_rate=0.01, weight_decay=0.001, adam_variant=optimizer_variant
            )
        )
        concept_owners = tuple(
            ParameterOptimizerState(
                parameters=tuple(training_batch.classifier.residual_adapter.parameters())
                + tuple(training_batch.classifier.classification_layer.parameters()),
                optimizer_settings=optimizer_settings,
            )
            for training_batch in training_batches
        )
        for model_id, training_batch, optimizer_owner in zip(
            (4, -7), training_batches, concept_owners
        ):
            register_training_state(registry, model_id, training_batch.classifier, optimizer_owner)
        legacy_registration_client.models = legacy_client.models
        legacy_client.model_training_examples = legacy_registration_client.model_training_examples
        legacy_client.model_optimizer_steps = legacy_registration_client.model_optimizer_steps
        legacy_training_batches = legacy_client._sample_training_batches()
        pending_upload_state = confirmation_arguments["pending_model_upload_state"]
        initial_parameter_snapshot = snapshot_classifier_parameters(
            classifier=training_batches[1].classifier
        )
        previous_parameter_snapshot = deepcopy(initial_parameter_snapshot)
        pending_upload_state.queue_model_upload(
            model_id=-7, parameter_snapshot=initial_parameter_snapshot, upload_delay_round_count=2
        )
        legacy_registration_client.pending_model_params = initial_parameter_snapshot
        pending_model_upload = pending_upload_state.get_pending_model_upload()
        previous_random_states = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        previous_numeric_environment = (
            torch.get_default_dtype(),
            torch.get_default_device(),
            torch.is_grad_enabled(),
        )
        for step_index in range(3):
            if step_index == 1:
                previous_states = registry.snapshot_ordered_held_model_training_states()
                previous_bindings = registry.snapshot_ordered_held_model_training_bindings()
                previous_optimizer_state = deepcopy(
                    concept_owners[1].parameter_optimizer.state_dict()
                )
                if optimizer_variant != "sgd":
                    assert previous_optimizer_state["state"]
                parameter_snapshots = snapshot_parameter_values_and_gradients(
                    parameter
                    for training_batch in training_batches
                    for parameter in training_batch.classifier.parameters()
                )
                assignment_change = confirm_held_model_registration(**confirmation_arguments)
                BaseClient.confirm_model_registration(legacy_registration_client, 12)
                legacy_training_batches[:] = [
                    (12 if model_id == -7 else model_id, input_features, observed_class_labels)
                    for model_id, input_features, observed_class_labels in legacy_training_batches
                ]
                assert_confirmation_matches_legacy(
                    confirmation_arguments,
                    legacy_registration_client,
                    compare_classifier_identity=False,
                )
                assert previous_states[1].model_id == previous_bindings[1].model_id == -7
                assert (
                    registry.get_held_model_training_state(model_id=12).classifier
                    is previous_states[1].classifier
                )
                assert (
                    registry.snapshot_ordered_held_model_training_bindings()[
                        1
                    ].concept_specific_parameter_optimizer
                    is previous_bindings[1].concept_specific_parameter_optimizer
                )
                assert_nested_state_equal(
                    concept_owners[1].parameter_optimizer.state_dict(), previous_optimizer_state
                )
                assert_parameter_values_and_gradients_unchanged(parameter_snapshots)
                assert (
                    assignment_change.previous_model_id == -7
                    and assignment_change.current_model_id == 12
                )
            training_bindings = registry.snapshot_ordered_held_model_training_bindings()
            participating_training_batches = tuple(
                replace(
                    training_batch,
                    classifier=training_binding.classifier,
                    concept_specific_parameter_optimizer=training_binding.concept_specific_parameter_optimizer,
                )
                for training_batch, training_binding in zip(training_batches, training_bindings)
            )
            expected_losses = run_legacy_joint_update(
                legacy_client=legacy_client, update_shared_features=update_shared_features
            )
            actual_losses = perform_joint_model_parameter_update(
                participating_training_batches=participating_training_batches,
                local_training_settings=LocalTrainingSettings(
                    local_model_parameter_update_strategy="joint_backbone_adapter_and_head_training",
                    shared_backbone_gradient_combination_strategy="sample_weighted_mean_per_concept_gradients",
                ),
                shared_feature_extractor=training_batches[0].classifier.feature_extractor,
                shared_parameter_optimizer=shared_optimizer,
                update_shared_features=update_shared_features,
            )
            assert actual_losses == expected_losses
            assert_joint_update_states_equal(
                participating_training_batches=participating_training_batches,
                shared_parameter_optimizer=shared_optimizer,
                legacy_client=legacy_client,
            )
            for training_binding, training_batch in zip(training_bindings, training_batches):
                confirmation_arguments[
                    "model_training_and_assignment_counts_store"
                ].record_completed_model_training(
                    model_id=training_binding.model_id,
                    trained_sample_count=len(training_batch.input_features),
                    parameter_update_step_count=1,
                )
            assert_model_counts_match_legacy(
                counts_store=confirmation_arguments["model_training_and_assignment_counts_store"],
                legacy_client=legacy_registration_client,
            )
        assert_confirmation_matches_legacy(
            confirmation_arguments, legacy_registration_client, compare_classifier_identity=False
        )
        assert pending_model_upload.model_id == -7
        assert pending_model_upload.parameter_snapshot is initial_parameter_snapshot
        assert_nested_state_equal(initial_parameter_snapshot, previous_parameter_snapshot)
        assert (
            training_batches[0].classifier.feature_extractor
            is training_batches[1].classifier.feature_extractor
        )
        assert random.getstate() == previous_random_states[0]
        assert np.random.get_state()[0] == previous_random_states[1][0]
        assert np.array_equal(np.random.get_state()[1], previous_random_states[1][1])
        assert np.random.get_state()[2:] == previous_random_states[1][2:]
        assert torch.equal(torch.get_rng_state(), previous_random_states[2])
        assert (
            torch.get_default_dtype(),
            torch.get_default_device(),
            torch.is_grad_enabled(),
        ) == previous_numeric_environment
