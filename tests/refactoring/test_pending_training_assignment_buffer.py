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

