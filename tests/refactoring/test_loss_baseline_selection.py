"""三用途の損失基準値を旧メソッドへ直接照合する。"""

from collections import defaultdict, deque
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from federated_drift_experiment import config
from federated_drift_experiment.clients.base import BaseClient
from federated_drift_experiment.clients.fedsda import ESRFedSDAClient, FedSDAClient
from federated_learning_experiments.learning.loss_statistics.bounded_loss_moments import (
    BoundedLossMoments,
)
from federated_learning_experiments.methods.fedsda.loss_statistics.loss_baseline_selection import (
    select_alarm_interval_reuse_baseline_mean_loss,
    select_loss_monitoring_baseline_mean_loss,
    select_post_alarm_reference_historical_mean_loss,
)


@pytest.mark.parametrize("observed_loss_count, mean_loss, sum_squared_loss_deviations",
    [(None, 0.0, 0.0), (0, 0.0, 0.0)] + [
        (observed_loss_count, mean_loss, 0.1 if observed_loss_count == 1 else 0.0)
        for observed_loss_count in (1, 2, 5)
        for mean_loss in (0.0, 0.005, 0.01, 0.2, 1.0 - 1e-6, 1.0 - 5e-7, 1.0)
    ])
def test_loss_baseline_matches_legacy_monitor(
    observed_loss_count, mean_loss, sum_squared_loss_deviations,
):
    legacy_stats = {} if observed_loss_count is None else {0: {
        "n": observed_loss_count, "mean": mean_loss, "M2": sum_squared_loss_deviations}}
    legacy_client = SimpleNamespace(current_model_id=0, model_stats=legacy_stats)
    loss_moments = None if observed_loss_count is None else BoundedLossMoments(
        observed_loss_count=observed_loss_count, mean_loss=mean_loss,
        sum_squared_loss_deviations=sum_squared_loss_deviations)
    state_before_call = None if loss_moments is None else vars(loss_moments).copy()
    result = select_loss_monitoring_baseline_mean_loss(loss_moments=loss_moments)
    assert type(result) is float
    assert result == ESRFedSDAClient._e_detector_baseline(legacy_client)
    assert (None if loss_moments is None else vars(loss_moments)) == state_before_call


@pytest.mark.parametrize("observed_loss_count, mean_loss, sum_squared_loss_deviations",
    [(None, 0.0, 0.0), (0, 0.0, 0.0)] + [
        (observed_loss_count, mean_loss, 0.1 if observed_loss_count == 1 else 0.0)
        for observed_loss_count in (1, 2, 5)
        for mean_loss in (0.0, 0.005, 0.01, 0.2, 1.0 - 1e-6, 1.0 - 5e-7, 1.0)
    ])
def test_loss_baseline_matches_legacy_reuse(
    observed_loss_count, mean_loss, sum_squared_loss_deviations, monkeypatch,
):
    monkeypatch.setattr(config, "MIN_DRIFT_DATA", 5)
    legacy_stats = {} if observed_loss_count is None else {0: {
        "n": observed_loss_count, "mean": mean_loss, "M2": sum_squared_loss_deviations}}
    legacy_client = SimpleNamespace(
        current_model_id=0, model_stats=legacy_stats, _forward_validation=None,
        buffer=deque((torch.zeros((1, 1)), torch.zeros((1, 1)), None) for _ in range(5)),
        _estimated_new_concept_span=lambda sample_idx: 5,
        verbose=False, distance_threshold=0.1,
        models={0: SimpleNamespace(per_sample_error=lambda bx, by: torch.zeros(len(bx)))},
        _record_model_compute=lambda *args: None,
        _select_reuse_candidate=Mock(side_effect=RuntimeError("oracle captured")),
        _select_initialization_params=Mock(side_effect=RuntimeError("oracle captured")),
    )
    legacy_client._get_model_stats = lambda model_id: BaseClient._get_model_stats(
        legacy_client, model_id)
    with pytest.raises(RuntimeError, match="oracle captured"):
        FedSDAClient._resolve_drift(legacy_client, sample_idx=10)
    if legacy_client._select_reuse_candidate.called:
        valid_candidates = legacy_client._select_reuse_candidate.call_args.args[0]
        assert valid_candidates == [(0, 0.0)]
        baseline_mean_loss = BaseClient._get_model_stats(legacy_client, 0)[0]
    else:
        evaluated_candidates = legacy_client._select_initialization_params.call_args.args[0]
        assert evaluated_candidates == []
        baseline_mean_loss = None
    loss_moments = None if observed_loss_count is None else BoundedLossMoments(
        observed_loss_count=observed_loss_count, mean_loss=mean_loss,
        sum_squared_loss_deviations=sum_squared_loss_deviations)
    state_before_call = None if loss_moments is None else vars(loss_moments).copy()
    result = select_alarm_interval_reuse_baseline_mean_loss(loss_moments=loss_moments)
    assert result == baseline_mean_loss
    assert result is None or type(result) is float
    assert (None if loss_moments is None else vars(loss_moments)) == state_before_call


@pytest.mark.parametrize("observed_loss_count, mean_loss, sum_squared_loss_deviations",
    [(None, 0.0, 0.0), (0, 0.0, 0.0)] + [
        (observed_loss_count, mean_loss, 0.1 if observed_loss_count == 1 else 0.0)
        for observed_loss_count in (1, 2, 5)
        for mean_loss in (0.0, 0.005, 0.01, 0.2, 1.0 - 1e-6, 1.0 - 5e-7, 1.0)
    ])
def test_loss_baseline_matches_legacy_history(
    observed_loss_count, mean_loss, sum_squared_loss_deviations, monkeypatch,
):
    monkeypatch.setattr(config, "NEW_MODEL_CREATION_POLICY", "forward_persistent")
    legacy_stats = {} if observed_loss_count is None else {0: {
        "n": observed_loss_count, "mean": mean_loss, "M2": sum_squared_loss_deviations}}
    candidate = SimpleNamespace(set_params=lambda initialization_params: None,
                                reset_optimizer=lambda: None)
    reference_models = {0: object()}
    legacy_client = SimpleNamespace(
        _new_model=lambda: candidate, _train_new_model=lambda *args: None,
        compute_counters=defaultdict(int), phase_seconds=defaultdict(float),
        _snapshot_reference_models=lambda: reference_models,
        model_stats=legacy_stats, current_model_id=0,
        _detector_label=lambda: "e-SR", forward_validation_samples=2,
    )
    FedSDAClient._begin_forward_validation(legacy_client,
        bx=torch.zeros((5, 1)), by=torch.zeros((5, 1)), drift_data=[],
        initialization_params={}, sample_idx=10, estimated_start=6, episode_id=None)
    captured_session = legacy_client._forward_validation
    baseline_mean_loss = captured_session.reference_historical_means.get(0)
    loss_moments = None if observed_loss_count is None else BoundedLossMoments(
        observed_loss_count=observed_loss_count, mean_loss=mean_loss,
        sum_squared_loss_deviations=sum_squared_loss_deviations)
    state_before_call = None if loss_moments is None else vars(loss_moments).copy()
    result = select_post_alarm_reference_historical_mean_loss(loss_moments=loss_moments)
    assert result == baseline_mean_loss
    assert result is None or type(result) is float
    assert (None if loss_moments is None else vars(loss_moments)) == state_before_call


@pytest.mark.parametrize("field_name,invalid_value", [
    ("loss_moments", True), ("loss_moments", 0), ("loss_moments", {}),
    ("loss_moments", []), ("loss_moments", "0.2"), ("loss_moments", object()),
    ("observed_loss_count", True), ("observed_loss_count", -1),
    ("observed_loss_count", 1.0), ("observed_loss_count", 10**400),
    ("mean_loss", True), ("mean_loss", float("nan")), ("mean_loss", -.1),
    ("mean_loss", 1.1), ("sum_squared_loss_deviations", -1),
    ("sum_squared_loss_deviations", float("inf")),
    ("sum_squared_loss_deviations", 10**400),
])
def test_loss_baseline_rejects_invalid_input(field_name, invalid_value):
    """改変された集計も検査し、拒否時は元の全fieldを保持する。"""
    loss_moments = BoundedLossMoments(observed_loss_count=2, mean_loss=.2, sum_squared_loss_deviations=.1)
    if field_name == "loss_moments":
        loss_moments = invalid_value
        state_before_call = None
    else:
        object.__setattr__(loss_moments, field_name, invalid_value)
        state_before_call = vars(loss_moments).copy()
    for operation in (select_loss_monitoring_baseline_mean_loss,
                      select_alarm_interval_reuse_baseline_mean_loss,
                      select_post_alarm_reference_historical_mean_loss):
        with pytest.raises((TypeError, ValueError), match=field_name):
            operation(loss_moments=loss_moments)
        if state_before_call is not None:
            assert all(vars(loss_moments)[field_name] is invalid_value
                       for field_name, invalid_value in state_before_call.items())


def test_loss_baseline_connects_to_monitor_and_reference():
    """監視のn1平均と履歴のn2平均0を異なるpublic用途へ渡す。"""
    from federated_learning_experiments.methods.fedsda.loss_change_detection.loss_change_detection_settings import LossChangeDetectionSettings
    from federated_learning_experiments.methods.fedsda.loss_change_detection.overall_and_true_class_loss_monitoring import OverallAndTrueClassLossMonitor
    from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_evaluation import select_available_reference_within_historical_loss_tolerance

    loss_moments = BoundedLossMoments(observed_loss_count=1, mean_loss=.2, sum_squared_loss_deviations=.1)
    baseline_mean_loss = select_loss_monitoring_baseline_mean_loss(loss_moments=loss_moments)
    assert select_alarm_interval_reuse_baseline_mean_loss(loss_moments=loss_moments) is None
    assert select_post_alarm_reference_historical_mean_loss(loss_moments=loss_moments) is None
    monitor = OverallAndTrueClassLossMonitor(
        loss_change_detection_settings=LossChangeDetectionSettings(
            drift_detector_name="e_sr", loss_monitoring_scope="overall_and_true_class_losses",
            e_sr_false_alarm_control_alpha=.01),
        class_count=2, initial_baseline_loss_mean=baseline_mean_loss,
        maximum_retained_candidate_count=5, betting_fractions=(.1, .4))
    observation = monitor.observe_loss_after_label_observation(
        observed_loss=.5, observed_class_id=0, sample_index=0,
        current_model_baseline_loss_mean=baseline_mean_loss)
    assert observation.component_update_count == 2
    assert monitor.get_state_snapshot().class_esr_states_by_class_id[0][1].baseline_loss_mean == .2
    loss_moments = BoundedLossMoments(observed_loss_count=2, mean_loss=0, sum_squared_loss_deviations=0)
    assert select_alarm_interval_reuse_baseline_mean_loss(loss_moments=loss_moments) is None
    reference_historical_mean_losses_by_model_id = {
        7: select_post_alarm_reference_historical_mean_loss(loss_moments=loss_moments)}
    reference_losses_by_model_id = {7: (.1, .2)}
    assert reference_historical_mean_losses_by_model_id == {7: 0.}
    assert select_available_reference_within_historical_loss_tolerance(
        reference_losses_by_model_id=reference_losses_by_model_id,
        reference_historical_mean_losses_by_model_id=reference_historical_mean_losses_by_model_id,
        available_reference_model_ids=(7,), current_training_model_id=7,
        maximum_reference_mean_loss_increase=.2) == 7


def test_loss_baseline_preserves_shared_state():
    """既定dtype/deviceと共有乱数、正常入力の表現を変更しない。"""
    import random
    import numpy as np
    import torch
    global_python_random_state = random.getstate()
    global_numpy_random_state = np.random.get_state()
    global_torch_random_state = torch.get_rng_state().clone()
    global_default_dtype = torch.get_default_dtype()
    global_default_device = torch.get_default_device()
    loss_moments = BoundedLossMoments(observed_loss_count=2, mean_loss=1, sum_squared_loss_deviations=0)
    object.__setattr__(loss_moments, "mean_loss", 1)
    state_before_call = vars(loss_moments).copy()
    try:
        torch.set_default_dtype(torch.float64)
        with torch.device("meta"):
            for operation in (select_loss_monitoring_baseline_mean_loss,
                              select_alarm_interval_reuse_baseline_mean_loss,
                              select_post_alarm_reference_historical_mean_loss):
                result = operation(loss_moments=loss_moments)
                assert type(result) is float
            assert torch.get_default_dtype() == torch.float64
            assert torch.get_default_device() == torch.device("meta")
    finally:
        torch.set_default_dtype(global_default_dtype)
    assert vars(loss_moments) == state_before_call
    assert type(loss_moments.mean_loss) is int
    assert random.getstate() == global_python_random_state
    assert np.random.get_state()[0] == global_numpy_random_state[0]
    assert np.array_equal(np.random.get_state()[1], global_numpy_random_state[1])
    assert np.random.get_state()[2:] == global_numpy_random_state[2:]
    assert torch.equal(torch.get_rng_state(), global_torch_random_state)
    assert torch.get_default_dtype() == global_default_dtype
    assert torch.get_default_device() == global_default_device


def test_loss_baseline_uses_keyword_arguments():
    """引数の役割はkeywordとして明示する。"""
    for operation in (select_loss_monitoring_baseline_mean_loss,
                      select_alarm_interval_reuse_baseline_mean_loss,
                      select_post_alarm_reference_historical_mean_loss):
        with pytest.raises(TypeError):
            operation(None)
