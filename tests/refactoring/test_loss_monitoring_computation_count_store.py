"""損失の監視（検出器）の計算の計数のowner: 加算、読取り、拒否。"""

from dataclasses import FrozenInstanceError

import pytest

from federated_learning_experiments.evaluation.loss_monitoring_computation_count_store import (
    LossMonitoringComputationCounts,
    LossMonitoringComputationCountStore,
)


def test_store_accumulates_recorded_counts():
    count_store = LossMonitoringComputationCountStore()
    assert count_store.get_loss_monitoring_computation_counts() == (
        LossMonitoringComputationCounts(
            detector_component_update_count=0, evaluated_candidate_bet_count=0
        )
    )
    assert (
        count_store.record_loss_monitoring_computation(
            detector_component_update_count=2, evaluated_candidate_bet_count=10
        )
        is None
    )
    counts_after_first_record = count_store.get_loss_monitoring_computation_counts()
    count_store.record_loss_monitoring_computation(
        detector_component_update_count=2, evaluated_candidate_bet_count=0
    )
    count_store.record_loss_monitoring_computation(
        detector_component_update_count=0, evaluated_candidate_bet_count=35
    )
    assert count_store.get_loss_monitoring_computation_counts() == (
        LossMonitoringComputationCounts(
            detector_component_update_count=4, evaluated_candidate_bet_count=45
        )
    )
    # 読取りは、後からの加算で変わらない。別のownerとは、計数を共有しない。
    assert counts_after_first_record == LossMonitoringComputationCounts(
        detector_component_update_count=2, evaluated_candidate_bet_count=10
    )
    with pytest.raises(FrozenInstanceError):
        counts_after_first_record.detector_component_update_count = 0
    assert LossMonitoringComputationCountStore().get_loss_monitoring_computation_counts() == (
        LossMonitoringComputationCounts(
            detector_component_update_count=0, evaluated_candidate_bet_count=0
        )
    )


@pytest.mark.parametrize(
    "count_name", ["detector_component_update_count", "evaluated_candidate_bet_count"]
)
@pytest.mark.parametrize(
    ("invalid_count", "expected_exception_type"),
    [(True, TypeError), (1.0, TypeError), ("1", TypeError), (None, TypeError), (-1, ValueError)],
)
def test_store_rejects_invalid_counts_without_changing_state(
    count_name, invalid_count, expected_exception_type
):
    count_store = LossMonitoringComputationCountStore()
    count_store.record_loss_monitoring_computation(
        detector_component_update_count=2, evaluated_candidate_bet_count=7
    )
    with pytest.raises(expected_exception_type):
        count_store.record_loss_monitoring_computation(
            **dict(detector_component_update_count=2, evaluated_candidate_bet_count=5)
            | {count_name: invalid_count}
        )
    assert count_store.get_loss_monitoring_computation_counts() == (
        LossMonitoringComputationCounts(
            detector_component_update_count=2, evaluated_candidate_bet_count=7
        )
    )


def test_record_requires_keyword_arguments():
    count_store = LossMonitoringComputationCountStore()
    with pytest.raises(TypeError):
        count_store.record_loss_monitoring_computation(2, 5)
