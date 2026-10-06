"""ID付替えの一覧順序・借用参照・学習状態を固定旧処理に照合する。"""

from copy import deepcopy

import numpy as np
import pytest
import torch
from test_held_model_training_state_registry import (
    build_registry_classifier_and_owner,
    register_training_state,
)
from test_joint_model_parameter_update import assert_nested_state_equal
from test_loss_statistics_model_id_reassignment import build_legacy_registration_client
from test_shared_feature_extractor_attachment import (
    assert_parameter_values_and_gradients_unchanged,
    snapshot_parameter_values_and_gradients,
)

from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatisticsStore,
)
from federated_learning_experiments.learning.training.held_model_training_state_registry import (
    HeldModelTrainingStateRegistry,
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
