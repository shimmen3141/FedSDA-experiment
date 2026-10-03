"""有界損失監視の単一数値状態と全体/class混合を旧基準へ照合する。"""

import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from federated_drift_experiment.drift_detectors.e_detector import BoundedMeanEDetector
from federated_learning_experiments.methods.fedsda.loss_change_detection.bounded_loss_e_sr_detection import (
    BoundedLossESRDetector,
)


def assert_single_series_matches_reference(*, detector, reference_detector):
    """公開結果と全候補capital/番号を丸めず比較する。"""
    state_snapshot = detector.get_state_snapshot()
    observation = detector.last_observation
    assert state_snapshot.baseline_loss_mean == reference_detector.baseline
    assert state_snapshot.candidate_log_capitals == tuple(tuple(log_values) for log_values in reference_detector._log_capitals.tolist())
    assert state_snapshot.candidate_start_observation_numbers == tuple(reference_detector._candidate_starts.tolist())
    assert observation.observed_loss_count == reference_detector._time
    assert observation.log_e_value == reference_detector.log_e_value
    assert observation.drift_detected == reference_detector.drift_detected
    assert observation.candidate_start_observation_number == reference_detector.split_start
    assert observation.oldest_retained_candidate_observation_number == reference_detector.retained_start_time
    assert observation.estimated_change_span_sample_count == reference_detector.width
    assert observation.evaluated_candidate_bet_count == reference_detector.active_hypothesis_count


@pytest.mark.parametrize("maximum_retained_candidate_count", [1, 7, 1000])
@pytest.mark.parametrize("baseline_loss_mean", [0.0, 0.2, 0.6, 1.0])
@pytest.mark.parametrize("betting_fractions", [(0.05, 0.1, 0.2, 0.4, 0.8), (0.4,), (0.4, 0.4)])
def test_loss_monitoring_single_series_matches_reference_after_each_observation(
    maximum_retained_candidate_count, baseline_loss_mean, betting_fractions,
):
    """低損失・上昇・回復・保持上限・baseline端点で全状態を照合する。"""
    detector = BoundedLossESRDetector(
        baseline_loss_mean=baseline_loss_mean, false_alarm_control_alpha=0.001,
        maximum_retained_candidate_count=maximum_retained_candidate_count, betting_fractions=betting_fractions,
    )
    reference_detector = BoundedMeanEDetector(baseline_loss_mean, 0.001, maximum_retained_candidate_count, betting_fractions)
    assert_single_series_matches_reference(detector=detector, reference_detector=reference_detector)
    for observed_loss in ([0.1] * 60 + [0.8] * 40 + [0.0, 1.0, 0.2] * 10):
        observation = detector.observe_loss(observed_loss=observed_loss)
        reference_detector.update(observed_loss)
        assert observation == detector.last_observation
        assert_single_series_matches_reference(detector=detector, reference_detector=reference_detector)


def test_loss_monitoring_candidate_ties_and_retention_match_reference():
    """候補平均にせず等号で警報とし、同率の最古候補を残す。"""
    detector = BoundedLossESRDetector(baseline_loss_mean=0.2, false_alarm_control_alpha=0.5, maximum_retained_candidate_count=3, betting_fractions=(0.1, 0.4))
    reference_detector = BoundedMeanEDetector(0.2, 0.5, 3, (0.1, 0.4))
    for observation_index in range(6):
        detector.observe_loss(observed_loss=0.2)
        reference_detector.update(0.2)
        assert_single_series_matches_reference(detector=detector, reference_detector=reference_detector)
        assert detector.last_observation.candidate_start_observation_number == max(1, observation_index - 1)
        if observation_index == 1:
            assert detector.last_observation.log_e_value == math.log(2)
            assert detector.last_observation.drift_detected


@pytest.mark.parametrize("baseline_loss_mean", [0.0, 0.01, 0.6, 1.0])
def test_loss_monitoring_reset_and_baseline_limits_match_reference(baseline_loss_mean):
    """resetは候補・時計・結果を消去し、指定baselineの安全域を保持する。"""
    detector = BoundedLossESRDetector(baseline_loss_mean=0.2, false_alarm_control_alpha=0.05, maximum_retained_candidate_count=2, betting_fractions=(0.1,))
    reference_detector = BoundedMeanEDetector(0.2, 0.05, 2, (0.1,))
    for observed_loss in (0.1, 1.0, 0.8):
        detector.observe_loss(observed_loss=observed_loss)
        reference_detector.update(observed_loss)
    state_snapshot = detector.get_state_snapshot()
    detector.reset(baseline_loss_mean=baseline_loss_mean)
    reference_detector.reset(baseline_loss_mean)
    assert_single_series_matches_reference(detector=detector, reference_detector=reference_detector)
    assert detector.last_observation.log_e_value == -math.inf
    assert state_snapshot.last_observation.observed_loss_count == 3
    detector.observe_loss(observed_loss=0.5)
    reference_detector.update(0.5)
    assert_single_series_matches_reference(detector=detector, reference_detector=reference_detector)


@pytest.mark.parametrize("invalid_parameter_name,invalid_parameter_value", [
    ('baseline_loss_mean', True), ('baseline_loss_mean', '0.2'), ('baseline_loss_mean', float('nan')),
    ('baseline_loss_mean', -0.1), ('baseline_loss_mean', 1.1), ('baseline_loss_mean', 10 ** 500),
    ('false_alarm_control_alpha', 0), ('false_alarm_control_alpha', 1), ('false_alarm_control_alpha', True),
    ('false_alarm_control_alpha', float('inf')),
    ('maximum_retained_candidate_count', True), ('maximum_retained_candidate_count', 0), ('maximum_retained_candidate_count', 1.5),
    ('betting_fractions', []), ('betting_fractions', ()), ('betting_fractions', (True,)),
    ('betting_fractions', (0.0,)), ('betting_fractions', (1.0,)), ('betting_fractions', (float('nan'),)),
])
def test_loss_monitoring_invalid_inputs_preserve_complete_state(invalid_parameter_name, invalid_parameter_value):
    """不正constructor条件を拒否し、不正loss/resetでは状態を保つ。"""
    detector_constructor_arguments = dict(baseline_loss_mean=0.2, false_alarm_control_alpha=0.05, maximum_retained_candidate_count=3, betting_fractions=(0.1, 0.4))
    with pytest.raises((TypeError, ValueError)) as exception_info:
        BoundedLossESRDetector(**(detector_constructor_arguments | {invalid_parameter_name: invalid_parameter_value}))
    assert str(exception_info.value)
    detector = BoundedLossESRDetector(**detector_constructor_arguments)
    detector.observe_loss(observed_loss=0.1)
    for invalid_parameter_value in (None, True, '0.2', float('nan'), float('inf'), -0.1, 1.1, 10 ** 500):
        state_before_call = detector.get_state_snapshot()
        with pytest.raises((TypeError, ValueError)):
            detector.observe_loss(observed_loss=invalid_parameter_value)
        assert detector.get_state_snapshot() == state_before_call
        with pytest.raises((TypeError, ValueError)):
            detector.reset(baseline_loss_mean=invalid_parameter_value)
        assert detector.get_state_snapshot() == state_before_call
    with pytest.raises(FrozenInstanceError):
        detector.last_observation.drift_detected = True
