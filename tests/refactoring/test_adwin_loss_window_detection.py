"""ADWINの検出器（単一の損失系列）: 旧の`FullScanADWIN`と、更新ごとに照合する。"""

import random

import pytest

from federated_drift_experiment import config
from federated_drift_experiment.drift_detectors.adwin import FullScanADWIN
from federated_learning_experiments.methods.fedsda.loss_change_detection.adwin_loss_window_detection import (
    AdwinLossWindowDetector,
    AdwinLossWindowObservation,
)


def assert_detector_matches_reference(*, detector, reference_detector):
    state_snapshot = detector.get_state_snapshot()
    assert state_snapshot.window_losses == tuple(reference_detector.window)
    assert state_snapshot.loss_sum == reference_detector.total
    assert state_snapshot.squared_loss_sum == reference_detector.total_sq
    assert state_snapshot.last_observation.window_sample_count == reference_detector.width
    assert state_snapshot.last_observation.drift_detected == reference_detector.drift_detected
    assert state_snapshot.last_observation is detector.last_observation


@pytest.mark.parametrize(
    ("confidence_delta", "maximum_window_sample_count", "minimum_test_window_sample_count"),
    ((0.05, 1000, 10), (0.002, 150, 10), (0.3, 60, 5)),
)
def test_adwin_detector_matches_reference_after_each_observation(
    confidence_delta, maximum_window_sample_count, minimum_test_window_sample_count, monkeypatch
):
    """窓・和・検知・調べた分割点の数が、損失1つごとに、旧の検出器と一致する（上限での切捨て、検知での削除を含む）。"""
    monkeypatch.setattr(config, "ADWIN_MIN_WIDTH", minimum_test_window_sample_count)
    detector = AdwinLossWindowDetector(
        confidence_delta=confidence_delta,
        maximum_window_sample_count=maximum_window_sample_count,
        minimum_test_window_sample_count=minimum_test_window_sample_count,
    )
    reference_detector = FullScanADWIN(
        delta=confidence_delta, max_window_size=maximum_window_sample_count
    )
    loss_generator = random.Random(17)
    observed_drift_values = set()
    window_was_trimmed_at_limit = False
    for observation_number in range(400):
        # 損失の水準を、何度か切り替える（検知と、検知の後の続きを通す）。
        loss_level = (0.1, 0.8, 0.3, 0.95)[(observation_number // 90) % 4]
        observed_loss = min(1.0, max(0.0, loss_level + loss_generator.uniform(-0.1, 0.1)))
        # 旧の計算量の計数と同じ式: 更新の前の窓+1（上限まで）が、検定を始める標本数以上なら、その数−1。
        scan_window_sample_count = min(reference_detector.width + 1, maximum_window_sample_count)
        window_was_trimmed_at_limit |= reference_detector.width == maximum_window_sample_count
        monitoring_observation = detector.observe_loss(observed_loss=observed_loss)
        reference_detector.update(observed_loss)
        assert type(monitoring_observation) is AdwinLossWindowObservation
        assert monitoring_observation.evaluated_split_count == (
            scan_window_sample_count - 1
            if scan_window_sample_count >= minimum_test_window_sample_count
            else 0
        )
        assert_detector_matches_reference(detector=detector, reference_detector=reference_detector)
        observed_drift_values.add(monitoring_observation.drift_detected)
    assert observed_drift_values == {False, True}
    # 窓が上限に達して、古い損失を捨てる経路は、上限の小さい条件で通る。
    assert window_was_trimmed_at_limit == (maximum_window_sample_count == 60)
    detector.reset()
    reference_detector.reset()
    assert_detector_matches_reference(detector=detector, reference_detector=reference_detector)
    assert detector.last_observation.evaluated_split_count == 0
    detector.observe_loss(observed_loss=0.5)
    reference_detector.update(0.5)
    assert_detector_matches_reference(detector=detector, reference_detector=reference_detector)


def test_adwin_detector_rejects_invalid_values_without_changing_state():
    for invalid_arguments, expected_exception in (
        (dict(confidence_delta=1), TypeError),
        (dict(confidence_delta=0.0), ValueError),
        (dict(confidence_delta=1.0), ValueError),
        (dict(maximum_window_sample_count=True), TypeError),
        (dict(maximum_window_sample_count=0), ValueError),
        (dict(minimum_test_window_sample_count=2.0), TypeError),
        (dict(minimum_test_window_sample_count=0), ValueError),
    ):
        with pytest.raises(expected_exception):
            AdwinLossWindowDetector(
                **dict(
                    confidence_delta=0.05,
                    maximum_window_sample_count=10,
                    minimum_test_window_sample_count=3,
                )
                | invalid_arguments
            )
    detector = AdwinLossWindowDetector(
        confidence_delta=0.05, maximum_window_sample_count=10, minimum_test_window_sample_count=3
    )
    detector.observe_loss(observed_loss=0.25)
    state_snapshot = detector.get_state_snapshot()
    for invalid_loss, expected_exception in (
        (True, TypeError),
        ("0.5", TypeError),
        (-0.1, ValueError),
        (1.5, ValueError),
        (float("nan"), ValueError),
    ):
        with pytest.raises(expected_exception):
            detector.observe_loss(observed_loss=invalid_loss)
    assert detector.get_state_snapshot() == state_snapshot
