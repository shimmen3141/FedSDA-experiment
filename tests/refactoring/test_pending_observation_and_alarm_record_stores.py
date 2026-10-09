"""保留標本のownerと、警報の記録のownerの、単独の動作を確かめる。"""

import math
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
import torch
from test_held_candidate_validation_progress import make_subclass_copy

from federated_learning_experiments.evaluation.loss_change_alarm_record_store import (
    LossChangeAlarmRecordSnapshot,
    LossChangeAlarmRecordStore,
)
from federated_learning_experiments.learning.training.indexed_observed_training_sample import (
    IndexedObservedTrainingSample,
)
from federated_learning_experiments.learning.training.model_training_sample_records import (
    ObservedTrainingSample,
)
from federated_learning_experiments.methods.fedsda.training_data_assignment.pending_sample_observation_store import (
    PendingSampleObservationStore,
)


class TupleSubclass(tuple):
    """exact tupleの検査が拒否するべき、tupleの派生型。"""


def make_pending_observation(sample_index):
    return IndexedObservedTrainingSample(
        sample_index=sample_index,
        training_sample=ObservedTrainingSample(
            input_features=torch.tensor([[float(sample_index), 0.5]]),
            observed_class_labels=torch.tensor([[0.0]]),
        ),
        observed_concept_id=sample_index % 2,
    )


def make_store_with_pending_observations(sample_indices):
    pending_sample_observation_store = PendingSampleObservationStore()
    pending_observations = tuple(
        make_pending_observation(sample_index) for sample_index in sample_indices
    )
    for pending_observation in pending_observations:
        pending_sample_observation_store.append_pending_sample_observation(
            indexed_observation=pending_observation
        )
    return pending_sample_observation_store, pending_observations


def test_pending_observation_store_keeps_appended_observations_in_order():
    pending_sample_observation_store, pending_observations = make_store_with_pending_observations(
        (5, 6, 9)
    )
    snapshot = pending_sample_observation_store.snapshot_pending_sample_observations()
    assert type(snapshot) is tuple
    assert len(snapshot) == 3
    assert all(
        stored_observation is pending_observation
        for stored_observation, pending_observation in zip(snapshot, pending_observations)
    )
    assert PendingSampleObservationStore().snapshot_pending_sample_observations() == ()
    assert pending_sample_observation_store.validation_assignment_sample_concept_ids is None


@pytest.mark.parametrize(
    "make_invalid_observation,expected_exception",
    [
        (lambda: object(), TypeError),
        (lambda: make_subclass_copy(make_pending_observation(10)), TypeError),
        (lambda: replace(make_pending_observation(10), sample_index=True), TypeError),
        (lambda: replace(make_pending_observation(10), sample_index=10.0), TypeError),
        # 保持中の最後の位置（9）以下の位置。
        (lambda: make_pending_observation(9), ValueError),
        (lambda: make_pending_observation(7), ValueError),
    ],
)
def test_pending_observation_store_rejects_invalid_observation_without_change(
    make_invalid_observation, expected_exception
):
    pending_sample_observation_store, pending_observations = make_store_with_pending_observations(
        (5, 6, 9)
    )
    with pytest.raises(expected_exception):
        pending_sample_observation_store.append_pending_sample_observation(
            indexed_observation=make_invalid_observation()
        )
    assert (
        pending_sample_observation_store.snapshot_pending_sample_observations()
        == pending_observations
    )


@pytest.mark.parametrize(
    "retained_sample_indices", [(), (9,), (6, 9), (5, 6, 9)], ids=["none", "one", "two", "all"]
)
def test_pending_observation_store_retains_latest_observations(retained_sample_indices):
    pending_sample_observation_store, pending_observations = make_store_with_pending_observations(
        (5, 6, 9)
    )
    pending_sample_observation_store.retain_latest_pending_sample_observations(
        retained_sample_indices=retained_sample_indices
    )
    snapshot = pending_sample_observation_store.snapshot_pending_sample_observations()
    assert snapshot == pending_observations[3 - len(retained_sample_indices) :]
    assert all(
        stored_observation is pending_observation
        for stored_observation, pending_observation in zip(
            snapshot, pending_observations[3 - len(retained_sample_indices) :]
        )
    )
    # 残した後も、最後の位置より後の標本を足せる（全部を外した後は、どの位置でも足せる）。
    pending_sample_observation_store.append_pending_sample_observation(
        indexed_observation=make_pending_observation(10 if retained_sample_indices else 0)
    )


@pytest.mark.parametrize(
    "retained_sample_indices,expected_exception",
    [
        ([6, 9], TypeError),
        (TupleSubclass((6, 9)), TypeError),
        # 古い側、途中を抜いた並び、逆順、保持より多い、保持にない位置。
        ((5, 6), ValueError),
        ((5, 9), ValueError),
        ((9, 6), ValueError),
        ((4, 5, 6, 9), ValueError),
        ((10,), ValueError),
    ],
)
def test_pending_observation_store_rejects_indices_that_are_not_the_latest(
    retained_sample_indices, expected_exception
):
    pending_sample_observation_store, pending_observations = make_store_with_pending_observations(
        (5, 6, 9)
    )
    with pytest.raises(expected_exception):
        pending_sample_observation_store.retain_latest_pending_sample_observations(
            retained_sample_indices=retained_sample_indices
        )
    assert (
        pending_sample_observation_store.snapshot_pending_sample_observations()
        == pending_observations
    )


def test_pending_observation_store_holds_validation_concept_ids_until_released():
    pending_sample_observation_store, pending_observations = make_store_with_pending_observations(
        (5, 6)
    )
    sample_concept_ids = (1, None, 0)
    pending_sample_observation_store.hold_validation_assignment_sample_concept_ids(
        sample_concept_ids=sample_concept_ids
    )
    assert (
        pending_sample_observation_store.validation_assignment_sample_concept_ids
        is sample_concept_ids
    )
    # 保持中の上書きは拒否し、保持している値を変えない。
    with pytest.raises(ValueError, match="already held"):
        pending_sample_observation_store.hold_validation_assignment_sample_concept_ids(
            sample_concept_ids=()
        )
    assert (
        pending_sample_observation_store.validation_assignment_sample_concept_ids
        is sample_concept_ids
    )
    # 保留標本の操作は、概念IDの保持に影響しない。
    pending_sample_observation_store.retain_latest_pending_sample_observations(
        retained_sample_indices=()
    )
    assert (
        pending_sample_observation_store.validation_assignment_sample_concept_ids
        is sample_concept_ids
    )
    assert (
        pending_sample_observation_store.release_validation_assignment_sample_concept_ids()
        is sample_concept_ids
    )
    assert pending_sample_observation_store.validation_assignment_sample_concept_ids is None
    with pytest.raises(ValueError, match="not held"):
        pending_sample_observation_store.release_validation_assignment_sample_concept_ids()
    # 空のtupleも「保持している」状態として扱う（候補検証へ渡した標本が0件）。
    pending_sample_observation_store.hold_validation_assignment_sample_concept_ids(
        sample_concept_ids=()
    )
    assert pending_sample_observation_store.validation_assignment_sample_concept_ids == ()
    assert pending_sample_observation_store.release_validation_assignment_sample_concept_ids() == ()


@pytest.mark.parametrize(
    "invalid_concept_ids", [[1, 0], TupleSubclass((1, 0)), (1, True), (1, 0.0), (1, "0"), None]
)
def test_pending_observation_store_rejects_invalid_concept_ids(invalid_concept_ids):
    pending_sample_observation_store = PendingSampleObservationStore()
    with pytest.raises(TypeError):
        pending_sample_observation_store.hold_validation_assignment_sample_concept_ids(
            sample_concept_ids=invalid_concept_ids
        )
    assert pending_sample_observation_store.validation_assignment_sample_concept_ids is None


def test_alarm_record_store_appends_values_and_alarm_positions():
    loss_change_alarm_record_store = LossChangeAlarmRecordStore()
    assert loss_change_alarm_record_store.get_state_snapshot() == LossChangeAlarmRecordSnapshot(
        monitored_log_e_values=(),
        alarm_sample_indices=(),
        estimated_change_point_sample_indices=(),
        detector_candidate_start_sample_indices=(),
    )
    for log_e_value in (-math.inf, -0.5, 3.25):
        loss_change_alarm_record_store.append_monitored_log_e_value(log_e_value=log_e_value)
    loss_change_alarm_record_store.append_alarm_record(
        alarm_sample_index=7,
        estimated_change_point_sample_index=5,
        detector_candidate_start_sample_index=2,
    )
    # 3つの位置が同じでもよい（警報の標本だけが変化区間）。
    loss_change_alarm_record_store.append_alarm_record(
        alarm_sample_index=8,
        estimated_change_point_sample_index=8,
        detector_candidate_start_sample_index=8,
    )
    state_snapshot = loss_change_alarm_record_store.get_state_snapshot()
    assert state_snapshot == LossChangeAlarmRecordSnapshot(
        monitored_log_e_values=(-math.inf, -0.5, 3.25),
        alarm_sample_indices=(7, 8),
        estimated_change_point_sample_indices=(5, 8),
        detector_candidate_start_sample_indices=(2, 8),
    )
    with pytest.raises(FrozenInstanceError):
        state_snapshot.alarm_sample_indices = ()
    # snapshotは、その後の追加の影響を受けない。
    loss_change_alarm_record_store.append_monitored_log_e_value(log_e_value=0.0)
    assert state_snapshot.monitored_log_e_values == (-math.inf, -0.5, 3.25)


@pytest.mark.parametrize(
    "invalid_log_e_value,expected_exception",
    [
        (1, TypeError),
        (True, TypeError),
        # numpyのfloat64はfloatの派生型。builtin floatだけを受け入れる。
        (np.float64(0.5), TypeError),
        ("0.5", TypeError),
        (math.nan, ValueError),
        (math.inf, ValueError),
    ],
)
def test_alarm_record_store_rejects_invalid_log_e_value(invalid_log_e_value, expected_exception):
    loss_change_alarm_record_store = LossChangeAlarmRecordStore()
    loss_change_alarm_record_store.append_monitored_log_e_value(log_e_value=0.25)
    state_snapshot = loss_change_alarm_record_store.get_state_snapshot()
    with pytest.raises(expected_exception):
        loss_change_alarm_record_store.append_monitored_log_e_value(log_e_value=invalid_log_e_value)
    assert loss_change_alarm_record_store.get_state_snapshot() == state_snapshot


@pytest.mark.parametrize(
    "invalid_alarm_record,expected_exception",
    [
        (dict(alarm_sample_index=True), TypeError),
        (dict(estimated_change_point_sample_index=5.0), TypeError),
        (dict(detector_candidate_start_sample_index=None), TypeError),
        # 候補開始≦変化区間の先頭≦警報、の順に反する位置。
        (dict(detector_candidate_start_sample_index=-1), ValueError),
        (dict(detector_candidate_start_sample_index=11), ValueError),
        (dict(estimated_change_point_sample_index=13), ValueError),
        # 記録済みの最後の警報（7）以下の警報の位置。
        (
            dict(
                alarm_sample_index=7,
                estimated_change_point_sample_index=7,
                detector_candidate_start_sample_index=7,
            ),
            ValueError,
        ),
    ],
)
def test_alarm_record_store_rejects_invalid_alarm_record(invalid_alarm_record, expected_exception):
    loss_change_alarm_record_store = LossChangeAlarmRecordStore()
    loss_change_alarm_record_store.append_alarm_record(
        alarm_sample_index=7,
        estimated_change_point_sample_index=5,
        detector_candidate_start_sample_index=2,
    )
    state_snapshot = loss_change_alarm_record_store.get_state_snapshot()
    with pytest.raises(expected_exception):
        loss_change_alarm_record_store.append_alarm_record(
            **dict(
                alarm_sample_index=12,
                estimated_change_point_sample_index=10,
                detector_candidate_start_sample_index=9,
            )
            | invalid_alarm_record
        )
    assert loss_change_alarm_record_store.get_state_snapshot() == state_snapshot
