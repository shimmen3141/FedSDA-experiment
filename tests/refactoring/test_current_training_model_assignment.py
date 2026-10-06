"""現在の単一学習帰属IDと、旧呼出し側の変更通知を対照する。"""

from dataclasses import FrozenInstanceError
from unittest.mock import Mock

import numpy as np
import pytest
from test_model_training_and_assignment_counts import (
    assert_model_counts_match_legacy,
    build_model_counts_oracle,
)

from federated_drift_experiment.clients.base import BaseClient
from federated_drift_experiment.clients.fedsda import FedSDAClient
from federated_learning_experiments.learning.training.current_training_model_assignment import (
    CurrentTrainingModelAssignment,
    TrainingModelAssignmentChange,
)


class IntSubclass(int):
    pass


class DictSubclass(dict):
    pass


def build_current_assignment_oracle(*, initial_model_id):
    _, legacy_client = build_model_counts_oracle()
    legacy_client.current_model_id = initial_model_id
    legacy_client._on_local_model_change = Mock()
    legacy_client.processed_samples = 37
    return CurrentTrainingModelAssignment(initial_model_id=initial_model_id), legacy_client


@pytest.mark.parametrize("initial_model_id", [0, -7, 4, 10**60])
@pytest.mark.parametrize("selected_model_id", [0, -7, 4, 10**60])
def test_local_assignment_matches_actual_legacy_hook(initial_model_id, selected_model_id):
    training_assignment, legacy_client = build_current_assignment_oracle(
        initial_model_id=initial_model_id
    )
    FedSDAClient._set_local_current_model(legacy_client, selected_model_id)
    assignment_change = training_assignment.assign_model_for_training(model_id=selected_model_id)
    assert training_assignment.current_training_model_id == legacy_client.current_model_id
    if selected_model_id == initial_model_id:
        assert assignment_change is None
        legacy_client._on_local_model_change.assert_not_called()
    else:
        assert assignment_change == TrainingModelAssignmentChange(
            previous_model_id=initial_model_id, current_model_id=selected_model_id
        )
        legacy_client._on_local_model_change.assert_called_once_with(
            assignment_change.previous_model_id, assignment_change.current_model_id
        )
    legacy_client._record_adaptation_event.assert_not_called()


@pytest.mark.parametrize("initial_model_id", [0, -7, 4, 10**60])
@pytest.mark.parametrize(
    "model_id_mapping",
    [{}, {4: 4}, {4: 8, 8: 12}, {4: -7, -7: 4}, {-7: 12}, {0: 4, 4: 8}, {10**60: 0}],
)
def test_mapping_matches_actual_legacy_id_and_server_event(initial_model_id, model_id_mapping):
    training_assignment, legacy_client = build_current_assignment_oracle(
        initial_model_id=initial_model_id
    )
    expected_model_id = model_id_mapping.get(initial_model_id, initial_model_id)
    BaseClient.apply_server_mapping(legacy_client, model_id_mapping, {})
    assignment_change = training_assignment.remap_current_training_model_id(
        model_id_mapping=model_id_mapping
    )
    assert training_assignment.current_training_model_id == expected_model_id
    assert legacy_client.current_model_id == expected_model_id
    legacy_client._on_local_model_change.assert_not_called()
    if expected_model_id == initial_model_id:
        assert assignment_change is None
        assert legacy_client.mapping_change_positions == []
        legacy_client._record_adaptation_event.assert_not_called()
    else:
        assert assignment_change == TrainingModelAssignmentChange(
            previous_model_id=initial_model_id, current_model_id=expected_model_id
        )
        assert legacy_client.mapping_change_positions == [37]
        legacy_client._record_adaptation_event.assert_called_once_with(
            position=37,
            detector="server",
            action="server_merge",
            old_model_id=assignment_change.previous_model_id,
            new_model_id=assignment_change.current_model_id,
        )


@pytest.mark.parametrize(
    "invalid_model_id", [None, True, False, 1.0, "4", IntSubclass(4), np.int64(4)]
)
def test_initial_id_rejects_implicit_conversion(invalid_model_id):
    with pytest.raises(TypeError):
        CurrentTrainingModelAssignment(initial_model_id=invalid_model_id)


@pytest.mark.parametrize(
    "invalid_model_id", [None, True, False, 1.0, "4", IntSubclass(4), np.int64(4)]
)
def test_assignment_rejection_preserves_current_id(invalid_model_id):
    training_assignment = CurrentTrainingModelAssignment(initial_model_id=-7)
    with pytest.raises(TypeError):
        training_assignment.assign_model_for_training(model_id=invalid_model_id)
    assert training_assignment.current_training_model_id == -7


@pytest.mark.parametrize(
    "invalid_mapping",
    [
        None,
        [],
        DictSubclass({-7: 12}),
        {-7: True},
        {True: 12},
        {-7: 1.0},
        {-7: IntSubclass(12)},
        {IntSubclass(4): 12},
        {-7: np.int64(12)},
        {np.int64(4): 12},
        {-7: 12, 999: "invalid"},
        {-7: 12, "invalid": 999},
    ],
)
def test_entire_mapping_is_validated_before_state_update(invalid_mapping):
    training_assignment = CurrentTrainingModelAssignment(initial_model_id=-7)
    with pytest.raises(TypeError):
        training_assignment.remap_current_training_model_id(model_id_mapping=invalid_mapping)
    assert training_assignment.current_training_model_id == -7


def test_readonly_id_and_retained_change_record_are_independent():
    training_assignment = CurrentTrainingModelAssignment(initial_model_id=-7)
    assignment_change = training_assignment.assign_model_for_training(model_id=12)
    assert assignment_change == TrainingModelAssignmentChange(
        previous_model_id=-7, current_model_id=12
    )
    with pytest.raises(AttributeError):
        training_assignment.current_training_model_id = 99
    with pytest.raises(FrozenInstanceError):
        assignment_change.current_model_id = 99
    training_assignment.assign_model_for_training(model_id=4)
    assert assignment_change.previous_model_id == -7
    assert assignment_change.current_model_id == 12
    assert training_assignment.current_training_model_id == 4


def test_mapping_input_is_neither_saved_nor_modified():
    training_assignment = CurrentTrainingModelAssignment(initial_model_id=-7)
    model_id_mapping = {-7: 12, 12: 99}
    assignment_change = training_assignment.remap_current_training_model_id(
        model_id_mapping=model_id_mapping
    )
    assert model_id_mapping == {-7: 12, 12: 99}
    model_id_mapping[12] = 4
    assert training_assignment.current_training_model_id == 12
    assert assignment_change.current_model_id == 12


@pytest.mark.parametrize("receiving_model_id", [0, 4, 12])
def test_registration_confirmation_connects_independent_id_and_count_owners(receiving_model_id):
    model_counts_store, legacy_client = build_model_counts_oracle(count_case="mixed")
    legacy_client.current_model_id = -7
    legacy_client.pending_model_params = {"snapshot": "independent"}
    # このtestでは新規モデル再構築を伴わない、保有モデルの登録確認を照合する。
    legacy_client.models[-7] = object()
    training_assignment = CurrentTrainingModelAssignment(initial_model_id=-7)
    expected_model_counts_snapshot = (
        model_counts_store.snapshot_model_training_and_assignment_counts()
    )
    assignment_change = training_assignment.assign_model_for_training(model_id=receiving_model_id)
    assert (
        model_counts_store.snapshot_model_training_and_assignment_counts()
        == expected_model_counts_snapshot
    )
    assert legacy_client.pending_model_params == {"snapshot": "independent"}
    assert legacy_client.current_model_id == -7
    assert -7 in legacy_client.models
    model_counts_store.transfer_model_training_and_assignment_counts(
        original_model_id=assignment_change.previous_model_id,
        receiving_model_id=assignment_change.current_model_id,
    )
    BaseClient.confirm_model_registration(legacy_client, receiving_model_id)
    assert training_assignment.current_training_model_id == legacy_client.current_model_id
    assert_model_counts_match_legacy(counts_store=model_counts_store, legacy_client=legacy_client)
    legacy_client._record_adaptation_event.assert_not_called()
    assert legacy_client.pending_model_params is None
    # 後続サーバ再編は両ownerに別々に適用する。返した登録recordは変更しない。
    model_id_mapping = {receiving_model_id: 99, 99: 100}
    model_counts_store.remap_model_training_and_assignment_counts(model_id_mapping=model_id_mapping)
    training_assignment.remap_current_training_model_id(model_id_mapping=model_id_mapping)
    BaseClient.apply_server_mapping(legacy_client, model_id_mapping, {})
    assert training_assignment.current_training_model_id == legacy_client.current_model_id == 99
    assert_model_counts_match_legacy(counts_store=model_counts_store, legacy_client=legacy_client)
    assert assignment_change.previous_model_id == -7
    assert assignment_change.current_model_id == receiving_model_id
