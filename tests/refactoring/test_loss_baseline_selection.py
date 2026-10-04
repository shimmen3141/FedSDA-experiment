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
