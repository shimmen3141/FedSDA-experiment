"""ID付替えの一覧順序・借用参照・学習状態を固定旧処理に照合する。"""

import random
from copy import deepcopy
from dataclasses import replace

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
from test_loss_statistics_model_id_reassignment import (
    assert_store_statistics_match_legacy,
    build_legacy_registration_client,
)
from test_shared_feature_extractor_attachment import (
    assert_parameter_values_and_gradients_unchanged,
    snapshot_parameter_values_and_gradients,
)

from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatisticsStore,
)
from federated_learning_experiments.learning.models.classifier_parameter_snapshot import (
    snapshot_classifier_parameters,
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


class DerivedModelId(int):
    """受入不可のint派生型。"""


@pytest.mark.parametrize("initial_model_ids", [(), (4,), (-7,), (-7, 4, 9), (4, -7, 9), (4, 9, -7)])
@pytest.mark.parametrize("reassigned_model_id", [4, 12, -7])
def test_training_state_id_reassignment_matches_legacy_registration(
    initial_model_ids, reassigned_model_id
):
    with torch.random.fork_rng(devices=[]):
        registry = HeldModelTrainingStateRegistry()
        legacy_registration_client = build_legacy_registration_client(
            loss_statistics_store=ModelAndClassLossStatisticsStore()
        )
        for model_id in initial_model_ids:
            classifier, optimizer_owner = build_registry_classifier_and_owner()
            register_training_state(registry, model_id, classifier, optimizer_owner)
            # 旧メソッドの値は新NN。実BaseClientのpop/代入だけを観測する。
            legacy_registration_client.models[model_id] = classifier
        previous_states = registry.snapshot_ordered_held_model_training_states()
        previous_bindings = registry.snapshot_ordered_held_model_training_bindings()
        registered_states_by_model_id = {state.model_id: state for state in previous_states}
        parameter_snapshots = snapshot_parameter_values_and_gradients(
            parameter for state in previous_states for parameter in state.classifier.parameters()
        )
        assert (
            registry.reassign_held_model_training_state_id(
                original_model_id=-7, reassigned_model_id=reassigned_model_id
            )
            is None
        )
        legacy_registration_client.confirm_model_registration(reassigned_model_id)
        assert tuple(
            state.model_id for state in registry.snapshot_ordered_held_model_training_states()
        ) == tuple(legacy_registration_client.models)
        for model_id, classifier in legacy_registration_client.models.items():
            reassigned_state = registry.get_held_model_training_state(model_id=model_id)
            source_state = registered_states_by_model_id[
                -7
                if model_id == reassigned_model_id and -7 in registered_states_by_model_id
                else model_id
            ]
            assert reassigned_state.classifier is classifier is source_state.classifier
            assert reassigned_state.concept_specific_parameter_optimizer_state is (
                source_state.concept_specific_parameter_optimizer_state
            )
        assert tuple(state.model_id for state in previous_states) == initial_model_ids
        assert tuple(binding.model_id for binding in previous_bindings) == initial_model_ids
        assert_parameter_values_and_gradients_unchanged(parameter_snapshots)


@pytest.mark.parametrize(
    "original_model_id,reassigned_model_id,expected_model_ids",
    [(0, -2, (19, -2)), (3, -4, (19, -4)), (3, 3, (19, 3)), (-5, -5, (19, -5))],
)
def test_training_state_id_reassignment_accepts_signed_ids(
    original_model_id, reassigned_model_id, expected_model_ids
):
    with torch.random.fork_rng(devices=[]):
        registry = HeldModelTrainingStateRegistry()
        classifier, optimizer_owner = build_registry_classifier_and_owner()
        for model_id in (original_model_id, 19):
            register_training_state(registry, model_id, classifier, optimizer_owner)
        previous_state = registry.get_held_model_training_state(model_id=original_model_id)
        registry.reassign_held_model_training_state_id(
            original_model_id=original_model_id, reassigned_model_id=reassigned_model_id
        )
        assert (
            tuple(
                state.model_id for state in registry.snapshot_ordered_held_model_training_states()
            )
            == expected_model_ids
        )
        reassigned_state = registry.get_held_model_training_state(model_id=reassigned_model_id)
        assert reassigned_state is not previous_state
        assert reassigned_state.classifier is classifier
        assert reassigned_state.concept_specific_parameter_optimizer_state is optimizer_owner
        assert previous_state.model_id == original_model_id


@pytest.mark.parametrize("invalid_model_id", [True, 1.0, "4", None, DerivedModelId(4), np.int64(4)])
@pytest.mark.parametrize("invalid_parameter_name", ["original_model_id", "reassigned_model_id"])
@pytest.mark.parametrize("source_present", [False, True])
@pytest.mark.parametrize("destination_present", [False, True])
def test_training_state_id_reassignment_rejects_invalid_ids_without_mutation(
    invalid_model_id, invalid_parameter_name, source_present, destination_present
):
    with torch.random.fork_rng(devices=[]):
        registry = HeldModelTrainingStateRegistry()
        classifier, optimizer_owner = build_registry_classifier_and_owner()
        for model_id in (
            *((-7,) if source_present else ()),
            *((4,) if destination_present else ()),
        ):
            register_training_state(registry, model_id, classifier, optimizer_owner)
        previous_states = registry.snapshot_ordered_held_model_training_states()
        parameter_snapshots = snapshot_parameter_values_and_gradients(classifier.parameters())
        previous_optimizer_state = deepcopy(optimizer_owner.parameter_optimizer.state_dict())
        with pytest.raises(ValueError, match=invalid_parameter_name):
            registry.reassign_held_model_training_state_id(
                original_model_id=(
                    invalid_model_id if invalid_parameter_name == "original_model_id" else -7
                ),
                reassigned_model_id=(
                    invalid_model_id if invalid_parameter_name == "reassigned_model_id" else 4
                ),
            )
        assert registry.snapshot_ordered_held_model_training_states() == previous_states
        assert_parameter_values_and_gradients_unchanged(parameter_snapshots)
        assert_nested_state_equal(
            optimizer_owner.parameter_optimizer.state_dict(), previous_optimizer_state
        )


def test_training_state_id_reassignment_preserves_old_records_and_current_optimizer():
    with torch.random.fork_rng(devices=[]):
        registry = HeldModelTrainingStateRegistry()
        classifier, optimizer_owner = build_registry_classifier_and_owner()
        for parameter in classifier.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer_owner.parameter_optimizer.step()
        register_training_state(registry, -7, classifier, optimizer_owner)
        previous_state = registry.get_held_model_training_state(model_id=-7)
        previous_binding = registry.snapshot_ordered_held_model_training_bindings()[0]
        previous_states = registry.snapshot_ordered_held_model_training_states()
        previous_optimizer = optimizer_owner.parameter_optimizer
        previous_optimizer_state = deepcopy(previous_optimizer.state_dict())
        assert previous_optimizer_state["state"]
        parameter_snapshots = snapshot_parameter_values_and_gradients(classifier.parameters())
        registry.reassign_held_model_training_state_id(original_model_id=-7, reassigned_model_id=12)
        with pytest.raises(KeyError) as invalid_case:
            registry.get_held_model_training_state(model_id=-7)
        assert invalid_case.value.args == (-7,)
        reassigned_state = registry.get_held_model_training_state(model_id=12)
        reassigned_binding = registry.snapshot_ordered_held_model_training_bindings()[0]
        assert reassigned_state is not previous_state
        assert reassigned_state.model_id == reassigned_binding.model_id == 12
        assert (
            previous_state.model_id
            == previous_binding.model_id
            == previous_states[0].model_id
            == -7
        )
        assert reassigned_state.classifier is previous_state.classifier is classifier
        assert reassigned_state.concept_specific_parameter_optimizer_state is optimizer_owner
        assert reassigned_binding.concept_specific_parameter_optimizer is previous_optimizer
        assert_nested_state_equal(previous_optimizer.state_dict(), previous_optimizer_state)
        optimizer_owner.reset_parameter_optimizer()
        reassigned_binding = registry.snapshot_ordered_held_model_training_bindings()[0]
        assert reassigned_binding.model_id == 12
        assert (
            reassigned_binding.concept_specific_parameter_optimizer
            is optimizer_owner.parameter_optimizer
        )
        assert reassigned_binding.concept_specific_parameter_optimizer is not previous_optimizer
        assert previous_binding.concept_specific_parameter_optimizer is previous_optimizer
        assert previous_state.concept_specific_parameter_optimizer_state is optimizer_owner
        assert_nested_state_equal(previous_optimizer.state_dict(), previous_optimizer_state)
        assert_parameter_values_and_gradients_unchanged(parameter_snapshots)


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize("optimizer_variant", ["standard", "amsgrad", "sgd"])
@pytest.mark.parametrize("update_shared_features", [True, False])
def test_reassigned_training_state_continues_joint_updates_and_statistics(
    class_count, optimizer_variant, update_shared_features, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        training_batches, shared_optimizer, legacy_client = build_joint_update_oracle_pair(
            class_count=class_count,
            batch_sample_counts=(3, 5),
            optimizer_variant=optimizer_variant,
            monkeypatch=monkeypatch,
        )
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
        registry = HeldModelTrainingStateRegistry()
        for model_id, training_batch, optimizer_owner in zip(
            (4, -7), training_batches, concept_owners
        ):
            register_training_state(registry, model_id, training_batch.classifier, optimizer_owner)
        loss_statistics_store = ModelAndClassLossStatisticsStore()
        loss_statistics_store.record_assigned_loss(
            model_id=-7, observed_loss=0.25, observed_class_id=0
        )
        legacy_registration_client = build_legacy_registration_client(
            loss_statistics_store=loss_statistics_store
        )
        legacy_registration_client.models = legacy_client.models
        legacy_registration_client.model_training_examples = legacy_client.model_training_examples
        legacy_registration_client.model_optimizer_steps = legacy_client.model_optimizer_steps
        legacy_training_batches = legacy_client._sample_training_batches()
        initial_parameter_snapshot = snapshot_classifier_parameters(
            classifier=training_batches[1].classifier
        )
        previous_parameter_snapshot = deepcopy(initial_parameter_snapshot)
        pending_upload_state = PendingModelUploadState()
        pending_upload_state.queue_model_upload(
            model_id=-7, parameter_snapshot=initial_parameter_snapshot, upload_delay_round_count=2
        )
        pending_model_upload = pending_upload_state.get_pending_model_upload()
        previous_random_states = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        previous_numeric_environment = (
            torch.get_default_dtype(),
            torch.get_default_device(),
            torch.is_grad_enabled(),
        )
        for step_index in range(3):
            if step_index == 1:
                previous_state = registry.get_held_model_training_state(model_id=-7)
                previous_binding = registry.snapshot_ordered_held_model_training_bindings()[1]
                previous_optimizer = concept_owners[1].parameter_optimizer
                previous_optimizer_state = deepcopy(previous_optimizer.state_dict())
                if optimizer_variant != "sgd":
                    assert previous_optimizer_state["state"]
                parameter_snapshots = snapshot_parameter_values_and_gradients(
                    parameter
                    for training_batch in training_batches
                    for parameter in training_batch.classifier.parameters()
                )
                current_loss_statistics = loss_statistics_store.get_model_loss_statistics(
                    model_id=-7
                )
                registry.reassign_held_model_training_state_id(
                    original_model_id=-7, reassigned_model_id=12
                )
                # 他ownerの移動は上位で別途行う。registry操作自体は統計/保留を変えない。
                assert (
                    loss_statistics_store.get_model_loss_statistics(model_id=-7)
                    == current_loss_statistics
                )
                assert loss_statistics_store.get_model_loss_statistics(model_id=12) is None
                assert pending_upload_state.get_pending_model_upload() is pending_model_upload
                assert pending_model_upload.model_id == -7
                assert pending_model_upload.parameter_snapshot is initial_parameter_snapshot
                assert pending_upload_state.remaining_upload_delay_round_count == 2
                assert not pending_upload_state.has_ready_model_upload()
                reassigned_state = registry.get_held_model_training_state(model_id=12)
                assert reassigned_state.classifier is previous_state.classifier
                assert (
                    reassigned_state.concept_specific_parameter_optimizer_state is concept_owners[1]
                )
                assert previous_state.model_id == previous_binding.model_id == -7
                assert (
                    registry.snapshot_ordered_held_model_training_bindings()[
                        1
                    ].concept_specific_parameter_optimizer
                    is previous_optimizer
                )
                assert_parameter_values_and_gradients_unchanged(parameter_snapshots)
                assert_nested_state_equal(previous_optimizer.state_dict(), previous_optimizer_state)
                loss_statistics_store.reassign_model_loss_statistics_id(
                    original_model_id=-7, reassigned_model_id=12
                )
                legacy_registration_client.confirm_model_registration(12)
                # sampler/storeのID対応はtest-only上位処理。標本と反復順は変えない。
                legacy_training_batches[:] = [
                    (12 if model_id == -7 else model_id, input_features, observed_class_labels)
                    for model_id, input_features, observed_class_labels in legacy_training_batches
                ]
                assert_store_statistics_match_legacy(
                    loss_statistics_store=loss_statistics_store,
                    legacy_client=legacy_registration_client,
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
        assert tuple(
            state.model_id for state in registry.snapshot_ordered_held_model_training_states()
        ) == (4, 12)
        assert pending_upload_state.get_pending_model_upload() is pending_model_upload
        assert_nested_state_equal(
            initial_parameter_snapshot,
            previous_parameter_snapshot,
        )
        assert (
            training_batches[0].classifier.feature_extractor
            is training_batches[1].classifier.feature_extractor
        )
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
