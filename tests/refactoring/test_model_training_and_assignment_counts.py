"""モデル別の診断件数を実旧加算・確認・対応へ照合する。"""

import random
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from test_held_model_joint_training_iterations import (
    build_training_iteration_oracle_pair,
    run_legacy_training_iterations,
)
from test_held_model_training_batch_sampling import assert_sampled_batches_equal
from test_joint_model_parameter_update import assert_joint_update_states_equal

from federated_drift_experiment.clients.base import BaseClient
from federated_learning_experiments.learning.training.held_model_joint_training_iterations import (
    perform_held_model_joint_training_iterations,
)
from federated_learning_experiments.learning.training.held_model_training_batch_sampling import (
    sample_training_batches_for_held_models,
)
from federated_learning_experiments.learning.training.local_training_settings import (
    LocalTrainingSettings,
)
from federated_learning_experiments.learning.training.model_training_and_assignment_counts import (
    ModelTrainingAndAssignmentCountsSnapshot,
    ModelTrainingAndAssignmentCountsStore,
)


class ModelIdIntSubclass(int):
    pass


class ModelIdMappingDictSubclass(dict):
    pass


def record_training_counts_in_both_implementations(
    *, counts_store, legacy_client, model_id, trained_sample_count, parameter_update_step_count
):
    legacy_client.compute_counters["training_examples"] = trained_sample_count
    legacy_client.compute_counters["optimizer_steps"] = parameter_update_step_count
    BaseClient._attribute_model_training(legacy_client, model_id, 0, 0)
    counts_store.record_completed_model_training(
        model_id=model_id,
        trained_sample_count=trained_sample_count,
        parameter_update_step_count=parameter_update_step_count,
    )


def record_concept_counts_in_both_implementations(
    *, counts_store, legacy_client, model_id, observed_concept_id
):
    BaseClient._record_model_concept(legacy_client, model_id, observed_concept_id)
    counts_store.record_assigned_sample_concept(
        model_id=model_id, observed_concept_id=observed_concept_id
    )


def build_model_counts_oracle(*, count_case="empty"):
    counts_store = ModelTrainingAndAssignmentCountsStore()
    legacy_client = SimpleNamespace(
        current_model_id=9999,
        models={},
        model_stats={},
        train_data_store=defaultdict(list),
        stored_data={},
        stored_data_limit=3,
        model_training_examples=defaultdict(int),
        model_optimizer_steps=defaultdict(int),
        model_concept_counts=defaultdict(Counter),
        compute_counters=defaultdict(int),
        pending_model_params=None,
        pending_model_stats=None,
        pending_model_ready=True,
        mapping_change_positions=[],
        processed_samples=0,
        _record_adaptation_event=Mock(),
        _after_models_rebuilt=Mock(),
    )
    if count_case in ("training_only", "mixed", "zero"):
        for model_id in (4, -7, 8):
            record_training_counts_in_both_implementations(
                counts_store=counts_store,
                legacy_client=legacy_client,
                model_id=model_id,
                trained_sample_count=0 if count_case == "zero" else model_id * model_id,
                parameter_update_step_count=0 if count_case == "zero" else 2,
            )
    if count_case in ("concept_only", "mixed"):
        for model_id, observed_concept_id in (
            (-7, 1),
            (-7, 2),
            (-7, 1),
            (4, 2),
            (4, -3),
            (9, 1),
            (8, None),
        ):
            record_concept_counts_in_both_implementations(
                counts_store=counts_store,
                legacy_client=legacy_client,
                model_id=model_id,
                observed_concept_id=observed_concept_id,
            )
    return counts_store, legacy_client


def assert_model_counts_match_legacy(*, counts_store, legacy_client):
    actual_snapshot = counts_store.snapshot_model_training_and_assignment_counts()
    assert type(actual_snapshot) is ModelTrainingAndAssignmentCountsSnapshot
    operation_arguments = (
        (actual_snapshot.trained_sample_counts_by_model_id, legacy_client.model_training_examples),
        (
            actual_snapshot.parameter_update_step_counts_by_model_id,
            legacy_client.model_optimizer_steps,
        ),
        (
            actual_snapshot.assigned_sample_counts_by_model_and_concept_id,
            legacy_client.model_concept_counts,
        ),
    )
    for actual_snapshot, expected_snapshot in operation_arguments:
        assert actual_snapshot == expected_snapshot
        assert tuple(actual_snapshot) == tuple(expected_snapshot)
        for model_id, increment in actual_snapshot.items():
            if isinstance(increment, dict):
                assert tuple(increment) == tuple(expected_snapshot[model_id])


@pytest.mark.parametrize("model_id", [-7, 0, 4])
@pytest.mark.parametrize("trained_sample_count", [0, 1, 10**80])
@pytest.mark.parametrize("parameter_update_step_count", [0, 2])
@pytest.mark.parametrize("observed_concept_id", [None, -3, 0, 2])
def test_model_counts_accumulation_matches_legacy(
    model_id, trained_sample_count, parameter_update_step_count, observed_concept_id
):
    counts_store, legacy_client = build_model_counts_oracle()
    for _ in range(2):
        record_training_counts_in_both_implementations(
            counts_store=counts_store,
            legacy_client=legacy_client,
            model_id=model_id,
            trained_sample_count=trained_sample_count,
            parameter_update_step_count=parameter_update_step_count,
        )
        record_concept_counts_in_both_implementations(
            counts_store=counts_store,
            legacy_client=legacy_client,
            model_id=model_id,
            observed_concept_id=observed_concept_id,
        )
        assert_model_counts_match_legacy(counts_store=counts_store, legacy_client=legacy_client)
        assert counts_store.get_model_assigned_sample_concept_counts(
            model_id=model_id
        ) == BaseClient.get_model_concept_counts(legacy_client, model_id)


@pytest.mark.parametrize("count_case", ["empty", "training_only", "concept_only", "mixed", "zero"])
@pytest.mark.parametrize("original_model_id", [-7, -99])
@pytest.mark.parametrize("receiving_model_id", [4, 12, -9])
def test_model_counts_transfer_matches_legacy(count_case, original_model_id, receiving_model_id):
    counts_store, legacy_client = build_model_counts_oracle(count_case=count_case)
    previous_snapshot = counts_store.snapshot_model_training_and_assignment_counts()
    legacy_client.current_model_id = original_model_id
    BaseClient.confirm_model_registration(legacy_client, receiving_model_id)
    counts_store.transfer_model_training_and_assignment_counts(
        original_model_id=original_model_id, receiving_model_id=receiving_model_id
    )
    assert_model_counts_match_legacy(counts_store=counts_store, legacy_client=legacy_client)
    if original_model_id in previous_snapshot.trained_sample_counts_by_model_id:
        assert (
            original_model_id
            not in counts_store.snapshot_model_training_and_assignment_counts().trained_sample_counts_by_model_id
        )
        assert previous_snapshot.trained_sample_counts_by_model_id[original_model_id] == (
            0 if count_case == "zero" else original_model_id**2
        )


@pytest.mark.parametrize("count_case", ["empty", "training_only", "concept_only", "mixed", "zero"])
@pytest.mark.parametrize(
    "model_id_mapping",
    [
        {},
        {4: 12},
        {-7: 4},
        {4: 8, 8: 12},
        {4: -7, -7: 4},
        {4: 12, -7: 12, 8: 12, 9: 12},
        {888: 12},
        {-7: -7, 4: 4},
    ],
)
def test_model_counts_one_hop_mapping_matches_legacy(count_case, model_id_mapping):
    counts_store, legacy_client = build_model_counts_oracle(count_case=count_case)
    for _ in range(2):
        BaseClient.apply_server_mapping(legacy_client, model_id_mapping, {})
        counts_store.remap_model_training_and_assignment_counts(model_id_mapping=model_id_mapping)
        assert_model_counts_match_legacy(counts_store=counts_store, legacy_client=legacy_client)


def test_model_counts_snapshots_and_getter_are_independent():
    counts_store, legacy_client = build_model_counts_oracle(count_case="mixed")
    previous_snapshot = counts_store.snapshot_model_training_and_assignment_counts()
    actual_snapshot = counts_store.snapshot_model_training_and_assignment_counts()
    actual_snapshot.trained_sample_counts_by_model_id.clear()
    actual_snapshot.parameter_update_step_counts_by_model_id[4] = -1
    actual_snapshot.assigned_sample_counts_by_model_and_concept_id[-7][1] = 999
    actual_snapshot.assigned_sample_counts_by_model_and_concept_id.clear()
    assert_model_counts_match_legacy(counts_store=counts_store, legacy_client=legacy_client)
    actual_snapshot = counts_store.get_model_assigned_sample_concept_counts(model_id=-7)
    actual_snapshot.clear()
    assert counts_store.get_model_assigned_sample_concept_counts(model_id=777) == {}
    assert counts_store.snapshot_model_training_and_assignment_counts() == previous_snapshot
    record_training_counts_in_both_implementations(
        counts_store=counts_store,
        legacy_client=legacy_client,
        model_id=4,
        trained_sample_count=2,
        parameter_update_step_count=1,
    )
    record_concept_counts_in_both_implementations(
        counts_store=counts_store, legacy_client=legacy_client, model_id=-7, observed_concept_id=1
    )
    assert previous_snapshot.trained_sample_counts_by_model_id[4] == 16
    assert previous_snapshot.assigned_sample_counts_by_model_and_concept_id[-7][1] == 2
    assert_model_counts_match_legacy(counts_store=counts_store, legacy_client=legacy_client)


def test_model_counts_rejects_same_id_before_legacy_corruption():
    counts_store, legacy_client = build_model_counts_oracle()
    record_training_counts_in_both_implementations(
        counts_store=counts_store,
        legacy_client=legacy_client,
        model_id=-7,
        trained_sample_count=3,
        parameter_update_step_count=2,
    )
    for _ in range(4):
        record_concept_counts_in_both_implementations(
            counts_store=counts_store,
            legacy_client=legacy_client,
            model_id=-7,
            observed_concept_id=1,
        )
    previous_snapshot = counts_store.snapshot_model_training_and_assignment_counts()
    legacy_client.current_model_id = -7
    BaseClient.confirm_model_registration(legacy_client, -7)
    assert dict(legacy_client.model_training_examples) == {-7: 6}
    assert dict(legacy_client.model_optimizer_steps) == {-7: 4}
    assert dict(legacy_client.model_concept_counts) == {}
    with pytest.raises(ValueError, match="receiving_model_id"):
        counts_store.transfer_model_training_and_assignment_counts(
            original_model_id=-7, receiving_model_id=-7
        )
    assert counts_store.snapshot_model_training_and_assignment_counts() == previous_snapshot


@pytest.mark.parametrize(
    "operation_name,invalid_parameter_name,invalid_value",
    [
        ("record_completed_model_training", "model_id", True),
        ("record_completed_model_training", "model_id", ModelIdIntSubclass(4)),
        ("record_completed_model_training", "trained_sample_count", True),
        ("record_completed_model_training", "trained_sample_count", np.int64(2)),
        ("record_completed_model_training", "trained_sample_count", -1),
        ("record_completed_model_training", "trained_sample_count", 1.0),
        ("record_completed_model_training", "parameter_update_step_count", None),
        ("record_completed_model_training", "parameter_update_step_count", -1),
        ("record_completed_model_training", "parameter_update_step_count", ModelIdIntSubclass(1)),
        ("record_assigned_sample_concept", "model_id", None),
        ("record_assigned_sample_concept", "observed_concept_id", True),
        ("record_assigned_sample_concept", "observed_concept_id", 1.5),
        ("record_assigned_sample_concept", "observed_concept_id", "1"),
        ("record_assigned_sample_concept", "observed_concept_id", np.int64(2)),
        ("record_assigned_sample_concept", "observed_concept_id", ModelIdIntSubclass(2)),
        ("transfer_model_training_and_assignment_counts", "original_model_id", True),
        ("transfer_model_training_and_assignment_counts", "receiving_model_id", np.int64(4)),
        ("remap_model_training_and_assignment_counts", "model_id_mapping", []),
        (
            "remap_model_training_and_assignment_counts",
            "model_id_mapping",
            ModelIdMappingDictSubclass(),
        ),
        ("remap_model_training_and_assignment_counts", "model_id_mapping", {4: 12, 8: True}),
        ("remap_model_training_and_assignment_counts", "model_id_mapping", {4: 12, "bad": 9}),
        ("get_model_assigned_sample_concept_counts", "model_id", True),
    ],
)
@pytest.mark.parametrize("count_case", ["empty", "mixed"])
def test_model_counts_rejects_invalid_inputs_atomically(
    operation_name, invalid_parameter_name, invalid_value, count_case
):
    counts_store, legacy_client = build_model_counts_oracle(count_case=count_case)
    previous_snapshot = counts_store.snapshot_model_training_and_assignment_counts()
    operation_arguments = {
        "record_completed_model_training": dict(
            model_id=4, trained_sample_count=3, parameter_update_step_count=2
        ),
        "record_assigned_sample_concept": dict(model_id=4, observed_concept_id=1),
        "transfer_model_training_and_assignment_counts": dict(
            original_model_id=-7, receiving_model_id=4
        ),
        "remap_model_training_and_assignment_counts": dict(model_id_mapping={4: 12}),
        "get_model_assigned_sample_concept_counts": dict(model_id=4),
    }[operation_name]
    operation_arguments[invalid_parameter_name] = invalid_value
    with pytest.raises((TypeError, ValueError), match=invalid_parameter_name):
        getattr(counts_store, operation_name)(**operation_arguments)
    assert counts_store.snapshot_model_training_and_assignment_counts() == previous_snapshot
    assert_model_counts_match_legacy(counts_store=counts_store, legacy_client=legacy_client)


def test_model_counts_validates_id_when_concept_is_unknown():
    counts_store = ModelTrainingAndAssignmentCountsStore()
    with pytest.raises(TypeError, match="model_id"):
        counts_store.record_assigned_sample_concept(model_id=True, observed_concept_id=None)
    with pytest.raises(ValueError, match="receiving_model_id"):
        counts_store.transfer_model_training_and_assignment_counts(
            original_model_id=99, receiving_model_id=99
        )
    assert (
        counts_store.snapshot_model_training_and_assignment_counts().assigned_sample_counts_by_model_and_concept_id
        == {}
    )


@pytest.mark.parametrize("original_model_id", [0, 4])
@pytest.mark.parametrize("receiving_model_id", [-2, 12])
def test_model_counts_supports_signed_distinct_id_transfer(original_model_id, receiving_model_id):
    counts_store = ModelTrainingAndAssignmentCountsStore()
    for model_id, trained_sample_count, parameter_update_step_count, observed_concept_id in (
        (original_model_id, 3, 2, 2),
        (receiving_model_id, 5, 1, 1),
    ):
        counts_store.record_completed_model_training(
            model_id=model_id,
            trained_sample_count=trained_sample_count,
            parameter_update_step_count=parameter_update_step_count,
        )
        counts_store.record_assigned_sample_concept(
            model_id=model_id, observed_concept_id=observed_concept_id
        )
    counts_store.transfer_model_training_and_assignment_counts(
        original_model_id=original_model_id, receiving_model_id=receiving_model_id
    )
    actual_snapshot = counts_store.snapshot_model_training_and_assignment_counts()
    assert actual_snapshot.trained_sample_counts_by_model_id == {receiving_model_id: 8}
    assert actual_snapshot.parameter_update_step_counts_by_model_id == {receiving_model_id: 3}
    assert tuple(
        actual_snapshot.assigned_sample_counts_by_model_and_concept_id[receiving_model_id].items()
    ) == ((1, 1), (2, 1))


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize("optimizer_variant", ["standard", "amsgrad", "sgd"])
@pytest.mark.parametrize("update_shared_features", [True, False])
def test_model_counts_connect_completed_training_and_id_transfer(
    class_count, optimizer_variant, update_shared_features, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        (
            training_batches,
            shared_optimizer,
            legacy_client,
            ordered_training_samples,
            training_bindings,
        ) = build_training_iteration_oracle_pair(
            class_count=class_count, optimizer_variant=optimizer_variant, monkeypatch=monkeypatch
        )
        counts_store, legacy_registration_client = build_model_counts_oracle()
        legacy_registration_client.current_model_id = -7
        legacy_registration_client.models = legacy_client.models
        legacy_registration_client.train_data_store = legacy_client.train_data_store
        legacy_registration_client.model_training_examples = legacy_client.model_training_examples
        legacy_registration_client.model_optimizer_steps = legacy_client.model_optimizer_steps
        legacy_client.model_concept_counts = legacy_registration_client.model_concept_counts
        for model_id, observed_concept_id in ((4, 0), (-7, 1), (-7, 1), (-7, None)):
            record_concept_counts_in_both_implementations(
                counts_store=counts_store,
                legacy_client=legacy_registration_client,
                model_id=model_id,
                observed_concept_id=observed_concept_id,
            )
        legacy_client.batch_size = 3
        python_random_generator = random.Random(731)
        previous_random_states = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        previous_numeric_environment = (
            torch.get_default_dtype(),
            torch.get_default_device(),
            torch.is_grad_enabled(),
        )
        for step_index in range(3):
            if step_index == 1:
                previous_snapshot = counts_store.snapshot_model_training_and_assignment_counts()
                BaseClient.confirm_model_registration(legacy_registration_client, 12)
                counts_store.transfer_model_training_and_assignment_counts(
                    original_model_id=-7, receiving_model_id=12
                )
                training_bindings = tuple(
                    replace(
                        training_binding,
                        model_id=12
                        if training_binding.model_id == -7
                        else training_binding.model_id,
                    )
                    for training_binding in training_bindings
                )
                ordered_training_samples = tuple(
                    replace(
                        training_sample_collection,
                        model_id=12
                        if training_sample_collection.model_id == -7
                        else training_sample_collection.model_id,
                    )
                    for training_sample_collection in ordered_training_samples
                )
                assert previous_snapshot.trained_sample_counts_by_model_id[-7] == 3
                assert previous_snapshot.parameter_update_step_counts_by_model_id[-7] == 1
                assert previous_snapshot.assigned_sample_counts_by_model_and_concept_id[-7] == {
                    1: 2
                }
            sampled_batches = sample_training_batches_for_held_models(
                held_model_ids=frozenset(
                    training_binding.model_id for training_binding in training_bindings
                ),
                ordered_model_training_samples=ordered_training_samples,
                batch_sample_count=3,
                python_random_generator=deepcopy(python_random_generator),
            )
            expected_losses, expected_random_state, sampled_batch_history = (
                run_legacy_training_iterations(
                    legacy_client=legacy_client,
                    iteration_count=1,
                    update_shared_features=update_shared_features,
                    initial_random_state=python_random_generator.getstate(),
                )
            )
            assert len(sampled_batch_history) == 1
            assert_sampled_batches_equal(
                sampled_batches=sampled_batches, legacy_batches=sampled_batch_history[0]
            )
            actual_losses = perform_held_model_joint_training_iterations(
                requested_joint_update_iteration_count=1,
                held_model_training_bindings=training_bindings,
                ordered_model_training_samples=ordered_training_samples,
                batch_sample_count=3,
                python_random_generator=python_random_generator,
                local_training_settings=LocalTrainingSettings(
                    local_model_parameter_update_strategy="joint_backbone_adapter_and_head_training",
                    shared_backbone_gradient_combination_strategy="sample_weighted_mean_per_concept_gradients",
                ),
                shared_feature_extractor=training_batches[0].classifier.feature_extractor,
                shared_parameter_optimizer=shared_optimizer,
                update_shared_features=update_shared_features,
            )
            assert actual_losses == expected_losses
            assert python_random_generator.getstate() == expected_random_state
            assert len(actual_losses) == 1
            for training_sample_collection in sampled_batches:
                counts_store.record_completed_model_training(
                    model_id=training_sample_collection.model_id,
                    trained_sample_count=len(training_sample_collection.input_features),
                    parameter_update_step_count=1,
                )
            assert_model_counts_match_legacy(counts_store=counts_store, legacy_client=legacy_client)
            assert_joint_update_states_equal(
                participating_training_batches=training_batches,
                shared_parameter_optimizer=shared_optimizer,
                legacy_client=legacy_client,
            )
        actual_snapshot = counts_store.snapshot_model_training_and_assignment_counts()
        assert actual_snapshot.trained_sample_counts_by_model_id == {4: 9, 12: 9}
        assert actual_snapshot.parameter_update_step_counts_by_model_id == {4: 3, 12: 3}
        assert actual_snapshot.assigned_sample_counts_by_model_and_concept_id == {
            4: {0: 1},
            12: {1: 2},
        }
        assert random.getstate() == previous_random_states[0]
        assert np.random.get_state()[0] == previous_random_states[1][0]
        assert np.array_equal(np.random.get_state()[1], previous_random_states[1][1])
        assert np.random.get_state()[2:] == previous_random_states[1][2:]
        assert torch.equal(torch.get_rng_state(), previous_random_states[2])
        assert previous_numeric_environment == (
            torch.get_default_dtype(),
            torch.get_default_device(),
            torch.is_grad_enabled(),
        )
