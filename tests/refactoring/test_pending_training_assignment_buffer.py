"""保留位置FIFOを旧clientの処理順・帰属順へ直接照合する。"""

from collections import defaultdict, deque
from dataclasses import FrozenInstanceError
import inspect
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from federated_drift_experiment import config
from federated_drift_experiment.clients.fedsda import FedSDAClient
from federated_learning_experiments.methods.fedsda.loss_change_detection.loss_change_detection_settings import LossChangeDetectionSettings
from federated_learning_experiments.methods.fedsda.loss_change_detection.overall_and_true_class_loss_monitoring import OverallAndTrueClassLossMonitor
from federated_learning_experiments.methods.fedsda.training_data_assignment.training_data_assignment_settings import TrainingDataAssignmentSettings
from federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer import PendingTrainingAssignmentBuffer


def make_assignment_buffer(capacity=3):
    """容量の所有者である固定条件を明示入力する。"""
    return PendingTrainingAssignmentBuffer(
        training_data_assignment_settings=TrainingDataAssignmentSettings(
            pending_assignment_buffer_capacity_samples=capacity,
        ),
    )


def make_legacy_processing_client(capacity, sample_index=0):
    """検出・学習をstubにし、旧process_one_stepのFIFO操作を実行する。"""
    return SimpleNamespace(
        processed_samples=sample_index, buffer=deque(), fifo_size=capacity,
        current_model_id=0, models={0: SimpleNamespace(get_absolute_error=lambda x, y: 0.0)},
        phase_seconds=defaultdict(float), processing_times=defaultdict(list),
        train_data_store={0: []}, history_drift_type=[],
        _record_prediction=lambda *x: None,
        _observe_forward_validation=lambda *x: 0,
        _record_model_compute=lambda *x: None,
        _update_drift_detectors=lambda *x: False,
        _forced_drift_check=lambda *x: False,
        _update_model_stats=lambda *x, **y: None,
        _record_model_concept=lambda *x: None,
        train_step=lambda: None,
    )


@pytest.mark.parametrize("capacity", [1, 3, 30])
@pytest.mark.parametrize("sample_index", [0, 71])
def test_pending_assignment_append_and_release_match_legacy_processing(capacity, sample_index):
    """追加直後のC+1保持と、平時の先頭解放順を旧処理に照合する。"""
    buffer = make_assignment_buffer(capacity)
    legacy_client = make_legacy_processing_client(capacity, sample_index)
    for observation_index in range(sample_index, sample_index + capacity + 5):
        buffer.append_observed_sample_index(sample_index=observation_index)
        state_snapshot = buffer.get_state_snapshot()
        assert state_snapshot.pending_sample_indices[-1] == observation_index
        assert len(state_snapshot.pending_sample_indices) == min(observation_index - sample_index + 1, capacity + 1)
        result = buffer.release_sample_indices_exceeding_capacity()
        FedSDAClient.process_one_step(
            legacy_client, torch.tensor([[float(observation_index)]]),
            torch.tensor([[0.0]]), observation_index,
        )
        legacy_sample_indices = tuple(int(x[0].item()) for x in legacy_client.train_data_store[0])
        assert legacy_sample_indices == tuple(range(sample_index, max(sample_index, observation_index - capacity + 1)))
        assert result == (() if observation_index < sample_index + capacity else (observation_index - capacity,))
        assert buffer.get_state_snapshot().pending_sample_indices == tuple(int(x[0].item()) for x in legacy_client.buffer)
        assert buffer.get_state_snapshot().last_observed_sample_index == observation_index
        assert buffer.release_sample_indices_exceeding_capacity() == ()


@pytest.mark.parametrize("invalid_value", [True, False, 1.0, "1", None, np.int64(1), -1, 2, 0])
def test_pending_assignment_invalid_inputs_preserve_state(invalid_value):
    """型・負値・順序違反を更新前に拒否する。"""
    buffer = make_assignment_buffer()
    buffer.append_observed_sample_index(sample_index=0)
    state_before_call = buffer.get_state_snapshot()
    with pytest.raises((TypeError, ValueError), match="sample_index"):
        buffer.append_observed_sample_index(sample_index=invalid_value)
    assert buffer.get_state_snapshot() == state_before_call


@pytest.mark.parametrize("invalid_value", [True, False, 1.0, "1", None, np.int64(1), 0, -1])
def test_pending_assignment_invalid_inputs_preserve_state_for_capacity(invalid_value):
    """改変された設定も既存検証器で再検査する。"""
    training_data_assignment_settings = TrainingDataAssignmentSettings(
        pending_assignment_buffer_capacity_samples=3,
    )
    object.__setattr__(training_data_assignment_settings, "pending_assignment_buffer_capacity_samples", invalid_value)
    with pytest.raises((TypeError, ValueError), match="pending_assignment_buffer_capacity_samples"):
        PendingTrainingAssignmentBuffer(training_data_assignment_settings=training_data_assignment_settings)


@pytest.mark.parametrize("invalid_value", [None, 3, {}, SimpleNamespace(pending_assignment_buffer_capacity_samples=3)])
def test_pending_assignment_invalid_inputs_preserve_state_for_settings_type(invalid_value):
    with pytest.raises(TypeError, match="training_data_assignment_settings"):
        PendingTrainingAssignmentBuffer(training_data_assignment_settings=invalid_value)


def test_pending_assignment_append_and_release_match_legacy_processing_for_large_indices():
    """位置は固定幅へ丸めず、初回は任意の非負位置を受け取る。"""
    buffer = make_assignment_buffer(1)
    sample_index = 10**30
    assert buffer.get_state_snapshot().pending_sample_indices == ()
    assert buffer.get_state_snapshot().last_observed_sample_index is None
    assert buffer.release_sample_indices_exceeding_capacity() == ()
    buffer.append_observed_sample_index(sample_index=sample_index)
    buffer.append_observed_sample_index(sample_index=sample_index + 1)
    buffer.append_observed_sample_index(sample_index=sample_index + 2)
    assert buffer.get_state_snapshot().pending_sample_indices == (sample_index, sample_index + 1, sample_index + 2)
    assert buffer.release_sample_indices_exceeding_capacity() == (sample_index, sample_index + 1)
    assert buffer.get_state_snapshot().pending_sample_indices == (sample_index + 2,)

def make_legacy_alarm_client(legacy_sample_indices, legacy_span, pending_validation=False):
    """整数tokenだけで旧警報の区間帰属とFIFO保持を捕捉する。"""
    legacy_events = {"assigned": [], "evaluated": [], "actions": [], "resets": []}
    legacy_client = SimpleNamespace(
        buffer=deque(legacy_sample_indices), current_model_id=0,
        _forward_validation=object() if pending_validation else None,
        _estimated_new_concept_span=lambda sample_index: legacy_span,
        _absorb_into_store=lambda model_id, x: legacy_events["assigned"].extend(x),
        _store_evaluation_data=lambda model_id, x: legacy_events["evaluated"].extend(x),
        _record_adaptation_event=lambda **x: legacy_events["actions"].append(x["action"]),
        _reset_drift_detectors=lambda: legacy_events["resets"].append(True),
        _detector_label=lambda: "test",
    )
    return legacy_client, legacy_events


@pytest.mark.parametrize("sample_count", [0, 1, 6, 30])
@pytest.mark.parametrize("legacy_span", [1, 2, 6, 80])
def test_pending_assignment_partition_matches_legacy_positions(sample_count, legacy_span):
    """FIFO内の切詰め開始と検出器本来の開始を区別する。"""
    buffer = make_assignment_buffer(30)
    for sample_index in range(101 - sample_count, 101):
        buffer.append_observed_sample_index(sample_index=sample_index)
    state_before_call = buffer.get_state_snapshot()
    result = buffer.get_change_interval_partition(estimated_change_span_sample_count=legacy_span)
    legacy_sample_indices = state_before_call.pending_sample_indices
    assert result.earlier_sample_indices + result.change_interval_sample_indices == legacy_sample_indices
    assert len(result.change_interval_sample_indices) == min(sample_count, legacy_span)
    assert result.change_interval_start_sample_index == (result.change_interval_sample_indices[0] if sample_count else None)
    assert buffer.get_state_snapshot() == state_before_call
    if sample_count:
        legacy_client, legacy_events = make_legacy_alarm_client(legacy_sample_indices, legacy_span)
        assert result.change_interval_start_sample_index == FedSDAClient._estimated_drift_start(legacy_client, 100)
        assert FedSDAClient._detector_candidate_start(legacy_client, 100) == max(0, 101 - legacy_span)
        assert result.change_interval_start_sample_index >= FedSDAClient._detector_candidate_start(legacy_client, 100)
        assert buffer.get_change_interval_partition(estimated_change_span_sample_count=legacy_span) == result


@pytest.mark.parametrize("operation", ["pending", "duplicate", "insufficient"])
@pytest.mark.parametrize("legacy_span", [1, 2, 8])
def test_pending_assignment_legacy_alarm_branches_preserve_consumption(monkeypatch, operation, legacy_span):
    """短い警報時の割当済みprefix残留を、今回も勝手に修正しない。"""
    monkeypatch.setattr(config, "MIN_DRIFT_DATA", 10)
    buffer = make_assignment_buffer(5)
    for sample_index in range(6):
        buffer.append_observed_sample_index(sample_index=sample_index)
    state_before_call = buffer.get_state_snapshot()
    legacy_client, legacy_events = make_legacy_alarm_client(
        state_before_call.pending_sample_indices, legacy_span, operation == "pending",
    )
    if operation == "duplicate":
        result = FedSDAClient._resolve_episode_duplicate(legacy_client, sample_idx=5, estimated_start=4, episode_id=1)
    else:
        result = FedSDAClient._resolve_drift(legacy_client, sample_idx=5, estimated_start=4, episode_id=1)
    assert result == 0
    assert legacy_events["resets"] == [True]
    if operation == "insufficient":
        result = buffer.get_change_interval_partition(estimated_change_span_sample_count=legacy_span)
        assert tuple(legacy_events["assigned"]) == result.earlier_sample_indices
        assert tuple(legacy_events["evaluated"]) == result.earlier_sample_indices
        assert legacy_events["actions"] == ["insufficient_data"]
        assert buffer.get_state_snapshot() == state_before_call
    else:
        result = buffer.drain_pending_sample_indices()
        assert tuple(legacy_events["assigned"]) == result == state_before_call.pending_sample_indices
        assert legacy_events["evaluated"] == []
        assert legacy_events["actions"] == [("forward_validation_pending" if operation == "pending" else "episode_suppressed")]
    assert buffer.get_state_snapshot().pending_sample_indices == tuple(legacy_client.buffer)
    assert buffer.get_state_snapshot().last_observed_sample_index == 5


@pytest.mark.parametrize("invalid_value", [True, False, 1.0, "1", None, np.int64(1), 0, -1])
def test_pending_assignment_invalid_inputs_preserve_state_for_span(invalid_value):
    buffer = make_assignment_buffer()
    buffer.append_observed_sample_index(sample_index=0)
    state_before_call = buffer.get_state_snapshot()
    with pytest.raises((TypeError, ValueError), match="estimated_change_span_sample_count"):
        buffer.get_change_interval_partition(estimated_change_span_sample_count=invalid_value)
    assert buffer.get_state_snapshot() == state_before_call


def test_pending_assignment_legacy_alarm_branches_preserve_consumption_after_drain():
    """消費後のindex reset・再観測を拒否し、全体の連続位置を維持する。"""
    buffer = make_assignment_buffer()
    assert buffer.drain_pending_sample_indices() == ()
    buffer.append_observed_sample_index(sample_index=71)
    buffer.append_observed_sample_index(sample_index=72)
    assert buffer.drain_pending_sample_indices() == (71, 72)
    assert buffer.drain_pending_sample_indices() == ()
    state_before_call = buffer.get_state_snapshot()
    assert state_before_call.pending_sample_indices == ()
    assert state_before_call.last_observed_sample_index == 72
    for invalid_value in (0, 71, 72, 74):
        with pytest.raises(ValueError, match="sample_index"):
            buffer.append_observed_sample_index(sample_index=invalid_value)
        assert buffer.get_state_snapshot() == state_before_call
    buffer.append_observed_sample_index(sample_index=73)
    assert buffer.get_change_interval_partition(estimated_change_span_sample_count=10**30).change_interval_sample_indices == (73,)

def test_pending_assignment_copies_instances_and_random_states_are_independent():
    """返却copyへの変更・別実体操作・共有乱数の副作用を検査する。"""
    global_python_random_state = random.getstate()
    global_numpy_random_state = np.random.get_state()
    global_torch_random_state = torch.get_rng_state().clone()
    buffer = make_assignment_buffer(1)
    reference_buffer = make_assignment_buffer(1)
    buffer.append_observed_sample_index(sample_index=71)
    state_snapshot = buffer.get_state_snapshot()
    result = buffer.get_change_interval_partition(estimated_change_span_sample_count=1)
    with pytest.raises(FrozenInstanceError):
        state_snapshot.last_observed_sample_index = 0
    with pytest.raises(FrozenInstanceError):
        state_snapshot.pending_sample_indices = ()
    with pytest.raises(FrozenInstanceError):
        result.change_interval_start_sample_index = 0
    for field_name in ("earlier_sample_indices", "change_interval_sample_indices"):
        with pytest.raises(FrozenInstanceError):
            setattr(result, field_name, ())
    buffer.append_observed_sample_index(sample_index=72)
    assert buffer.release_sample_indices_exceeding_capacity() == (71,)
    assert buffer.drain_pending_sample_indices() == (72,)
    reference_buffer.append_observed_sample_index(sample_index=0)
    assert state_snapshot.pending_sample_indices == (71,)
    assert result.change_interval_sample_indices == (71,)
    assert reference_buffer.get_state_snapshot().pending_sample_indices == (0,)
    assert buffer.get_state_snapshot().pending_sample_indices == ()
    assert random.getstate() == global_python_random_state
    assert np.random.get_state()[0] == global_numpy_random_state[0]
    np.testing.assert_array_equal(np.random.get_state()[1], global_numpy_random_state[1])
    assert np.random.get_state()[2:] == global_numpy_random_state[2:]
    assert torch.equal(torch.get_rng_state(), global_torch_random_state)


@pytest.mark.parametrize("capacity", [1, 3, 30])
def test_pending_assignment_monitoring_span_connects_to_buffer_partition(capacity):
    """monitor public結果を入力し、FIFO前の候補を明示的に切り詰める。"""
    loss_change_detection_settings = LossChangeDetectionSettings(
        e_sr_false_alarm_control_alpha=.01,
        drift_detector_name="e_sr",
        loss_monitoring_scope="overall_and_true_class_losses",
    )
    monitor = OverallAndTrueClassLossMonitor(
        loss_change_detection_settings=loss_change_detection_settings,
        class_count=2, initial_baseline_loss_mean=.2,
        maximum_retained_candidate_count=80, betting_fractions=(.2, .5, .8),
    )
    buffer = make_assignment_buffer(capacity)
    for observation_index in range(101):
        observation = monitor.observe_loss_after_label_observation(
            observed_loss=(.1 if observation_index < 30 else .9),
            observed_class_id=observation_index % 2, sample_index=observation_index,
            current_model_baseline_loss_mean=.2,
        )
        buffer.append_observed_sample_index(sample_index=observation_index)
        result = buffer.get_change_interval_partition(
            estimated_change_span_sample_count=observation.estimated_change_span_sample_count,
        )
        state_snapshot = buffer.get_state_snapshot()
        assert result.change_interval_start_sample_index == max(
            state_snapshot.pending_sample_indices[0],
            observation.detector_candidate_start_sample_index,
        )
        assert result.change_interval_sample_indices[-1] == observation.sample_index
        assert buffer.get_state_snapshot() == state_snapshot
        buffer.release_sample_indices_exceeding_capacity()
    assert observation.detector_candidate_start_sample_index < result.change_interval_start_sample_index
    # 最終結果は容量解放前の参照で、警報操作をbufferが自動選択しない。
    assert buffer.get_state_snapshot().last_observed_sample_index == 100


def test_pending_assignment_public_functions_require_explicit_keyword_arguments():
    for operation, field_name in (
        (PendingTrainingAssignmentBuffer, "training_data_assignment_settings"),
        (PendingTrainingAssignmentBuffer.append_observed_sample_index, "sample_index"),
        (PendingTrainingAssignmentBuffer.get_change_interval_partition, "estimated_change_span_sample_count"),
    ):
        assert inspect.signature(operation).parameters[field_name].kind is inspect.Parameter.KEYWORD_ONLY
    buffer = make_assignment_buffer()
    with pytest.raises(TypeError):
        buffer.append_observed_sample_index(0)
    with pytest.raises(TypeError):
        buffer.get_change_interval_partition(1)
    with pytest.raises(TypeError):
        PendingTrainingAssignmentBuffer(TrainingDataAssignmentSettings(pending_assignment_buffer_capacity_samples=3))
