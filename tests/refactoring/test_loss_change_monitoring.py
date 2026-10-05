"""有界損失監視の単一数値状態と全体/class混合を旧基準へ照合する。"""

import inspect
import math
import random
from collections import defaultdict, deque
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
import torch
from test_run_settings_validation import valid_run_settings_mapping as valid_run_settings_mapping

from federated_drift_experiment import config
from federated_drift_experiment.clients.fedsda import (
    ClassConditionalESRFedSDAClient,
    _AdaHedgeRoutingFedSDAClientMixin,
)
from federated_drift_experiment.drift_detectors.e_detector import BoundedMeanEDetector
from federated_learning_experiments.learning.prediction.class_probability_calculations import (
    compute_model_mean_bounded_losses_after_label_observation,
    convert_model_outputs_to_prediction_probabilities,
)
from federated_learning_experiments.methods.fedsda.loss_change_detection.bounded_loss_e_sr_detection import (
    BoundedLossESRDetector,
    BoundedLossESRObservation,
)
from federated_learning_experiments.methods.fedsda.loss_change_detection.overall_and_true_class_loss_monitoring import (
    OverallAndTrueClassLossMonitor,
)


def assert_single_series_matches_reference(*, detector, reference_detector):
    """公開結果と全候補capital/番号を丸めず比較する。"""
    state_snapshot = detector.get_state_snapshot()
    observation = detector.last_observation
    assert state_snapshot.baseline_loss_mean == reference_detector.baseline
    assert state_snapshot.candidate_log_capitals == tuple(
        tuple(log_values) for log_values in reference_detector._log_capitals.tolist()
    )
    assert state_snapshot.candidate_start_observation_numbers == tuple(
        reference_detector._candidate_starts.tolist()
    )
    assert observation.observed_loss_count == reference_detector._time
    assert observation.log_e_value == reference_detector.log_e_value
    assert observation.drift_detected == reference_detector.drift_detected
    assert observation.candidate_start_observation_number == reference_detector.split_start
    assert (
        observation.oldest_retained_candidate_observation_number
        == reference_detector.retained_start_time
    )
    assert observation.estimated_change_span_sample_count == reference_detector.width
    assert observation.evaluated_candidate_bet_count == reference_detector.active_hypothesis_count


@pytest.mark.parametrize("maximum_retained_candidate_count", [1, 7, 1000])
@pytest.mark.parametrize("baseline_loss_mean", [0.0, 0.2, 0.6, 1.0])
@pytest.mark.parametrize("betting_fractions", [(0.05, 0.1, 0.2, 0.4, 0.8), (0.4,), (0.4, 0.4)])
def test_loss_monitoring_single_series_matches_reference_after_each_observation(
    maximum_retained_candidate_count,
    baseline_loss_mean,
    betting_fractions,
):
    """低損失・上昇・回復・保持上限・baseline端点で全状態を照合する。"""
    detector = BoundedLossESRDetector(
        baseline_loss_mean=baseline_loss_mean,
        false_alarm_control_alpha=0.001,
        maximum_retained_candidate_count=maximum_retained_candidate_count,
        betting_fractions=betting_fractions,
    )
    reference_detector = BoundedMeanEDetector(
        baseline_loss_mean, 0.001, maximum_retained_candidate_count, betting_fractions
    )
    assert_single_series_matches_reference(detector=detector, reference_detector=reference_detector)
    for observed_loss in [0.1] * 60 + [0.8] * 40 + [0.0, 1.0, 0.2] * 10:
        observation = detector.observe_loss(observed_loss=observed_loss)
        reference_detector.update(observed_loss)
        assert observation == detector.last_observation
        assert_single_series_matches_reference(
            detector=detector, reference_detector=reference_detector
        )


def test_loss_monitoring_candidate_ties_and_retention_match_reference():
    """候補平均にせず等号で警報とし、同率の最古候補を残す。"""
    detector = BoundedLossESRDetector(
        baseline_loss_mean=0.2,
        false_alarm_control_alpha=0.5,
        maximum_retained_candidate_count=3,
        betting_fractions=(0.1, 0.4),
    )
    reference_detector = BoundedMeanEDetector(0.2, 0.5, 3, (0.1, 0.4))
    for observation_index in range(6):
        detector.observe_loss(observed_loss=0.2)
        reference_detector.update(0.2)
        assert_single_series_matches_reference(
            detector=detector, reference_detector=reference_detector
        )
        assert detector.last_observation.candidate_start_observation_number == max(
            1, observation_index - 1
        )
        if observation_index == 1:
            assert detector.last_observation.log_e_value == math.log(2)
            assert detector.last_observation.drift_detected


@pytest.mark.parametrize("baseline_loss_mean", [0.0, 0.01, 0.6, 1.0])
def test_loss_monitoring_reset_and_baseline_limits_match_reference(baseline_loss_mean):
    """resetは候補・時計・結果を消去し、指定baselineの安全域を保持する。"""
    detector = BoundedLossESRDetector(
        baseline_loss_mean=0.2,
        false_alarm_control_alpha=0.05,
        maximum_retained_candidate_count=2,
        betting_fractions=(0.1,),
    )
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


@pytest.mark.parametrize(
    "invalid_parameter_name,invalid_parameter_value",
    [
        ("baseline_loss_mean", True),
        ("baseline_loss_mean", "0.2"),
        ("baseline_loss_mean", float("nan")),
        ("baseline_loss_mean", -0.1),
        ("baseline_loss_mean", 1.1),
        ("baseline_loss_mean", 10**500),
        ("false_alarm_control_alpha", 0),
        ("false_alarm_control_alpha", 1),
        ("false_alarm_control_alpha", True),
        ("false_alarm_control_alpha", float("inf")),
        ("maximum_retained_candidate_count", True),
        ("maximum_retained_candidate_count", 0),
        ("maximum_retained_candidate_count", 1.5),
        ("betting_fractions", []),
        ("betting_fractions", ()),
        ("betting_fractions", (True,)),
        ("betting_fractions", (0.0,)),
        ("betting_fractions", (1.0,)),
        ("betting_fractions", (float("nan"),)),
    ],
)
def test_loss_monitoring_invalid_inputs_preserve_complete_state(
    invalid_parameter_name, invalid_parameter_value
):
    """不正constructor条件を拒否し、不正loss/resetでは状態を保つ。"""
    detector_constructor_arguments = dict(
        baseline_loss_mean=0.2,
        false_alarm_control_alpha=0.05,
        maximum_retained_candidate_count=3,
        betting_fractions=(0.1, 0.4),
    )
    with pytest.raises((TypeError, ValueError)) as exception_info:
        BoundedLossESRDetector(
            **(detector_constructor_arguments | {invalid_parameter_name: invalid_parameter_value})
        )
    assert str(exception_info.value)
    detector = BoundedLossESRDetector(**detector_constructor_arguments)
    detector.observe_loss(observed_loss=0.1)
    for invalid_parameter_value in (
        None,
        True,
        "0.2",
        float("nan"),
        float("inf"),
        -0.1,
        1.1,
        10**500,
    ):
        state_before_call = detector.get_state_snapshot()
        with pytest.raises((TypeError, ValueError)):
            detector.observe_loss(observed_loss=invalid_parameter_value)
        assert detector.get_state_snapshot() == state_before_call
        with pytest.raises((TypeError, ValueError)):
            detector.reset(baseline_loss_mean=invalid_parameter_value)
        assert detector.get_state_snapshot() == state_before_call
    with pytest.raises(FrozenInstanceError):
        detector.last_observation.drift_detected = True


def make_reference_class_monitor(
    *,
    monkeypatch,
    class_count,
    baseline_loss_mean,
    maximum_retained_candidate_count,
    false_alarm_control_alpha,
):
    """モデル生成を伴わず旧ClassESR実メソッドを呼ぶtest専用fixture。"""
    monkeypatch.setattr(config, "num_classes", lambda: class_count)
    monkeypatch.setattr(config, "E_DETECTOR_ALPHA", false_alarm_control_alpha)
    monkeypatch.setattr(config, "ADWIN_MAX_WINDOW", maximum_retained_candidate_count)
    reference_monitor = ClassConditionalESRFedSDAClient.__new__(ClassConditionalESRFedSDAClient)
    reference_monitor.current_model_id = 0
    reference_monitor.model_stats = {0: {"n": 1, "mean": baseline_loss_mean}}
    reference_monitor.e_detector = BoundedMeanEDetector(
        baseline_loss_mean, false_alarm_control_alpha, maximum_retained_candidate_count
    )
    reference_monitor.overall_component_weight = 1.0 / (class_count + 1)
    reference_monitor.class_component_weight = (
        1.0 - reference_monitor.overall_component_weight
    ) / class_count
    reference_monitor.class_e_detectors = {}
    reference_monitor.class_e_positions = defaultdict(deque)
    reference_monitor._class_drift_start = None
    reference_monitor.history_detector_log_e = []
    reference_monitor.compute_counters = {
        "drift_detector_updates": 0,
        "drift_detector_hypotheses": 0,
    }
    return reference_monitor


def assert_class_monitor_matches_reference(*, monitor, reference_monitor):
    """public snapshotから全成分capitalとclass位置を照合する。"""
    state_snapshot = monitor.get_state_snapshot()
    assert tuple(
        class_id for class_id, reference_state in state_snapshot.class_esr_states_by_class_id
    ) == tuple(reference_monitor.class_e_detectors)
    assert state_snapshot.class_sample_indices_by_class_id == tuple(
        (class_id, tuple(class_sample_indices))
        for class_id, class_sample_indices in reference_monitor.class_e_positions.items()
    )
    for class_id, detector_state_snapshot in (
        (None, state_snapshot.overall_esr_state),
        *state_snapshot.class_esr_states_by_class_id,
    ):
        reference_detector = (
            reference_monitor.e_detector
            if class_id is None
            else reference_monitor.class_e_detectors[class_id]
        )
        assert detector_state_snapshot.baseline_loss_mean == reference_detector.baseline
        assert detector_state_snapshot.candidate_log_capitals == tuple(
            tuple(log_values) for log_values in reference_detector._log_capitals.tolist()
        )
        assert detector_state_snapshot.candidate_start_observation_numbers == tuple(
            reference_detector._candidate_starts.tolist()
        )
        reference_observation = BoundedLossESRObservation(
            observed_loss_count=reference_detector._time,
            log_e_value=reference_detector.log_e_value,
            drift_detected=reference_detector.drift_detected,
            candidate_start_observation_number=reference_detector.split_start,
            oldest_retained_candidate_observation_number=reference_detector.retained_start_time,
            estimated_change_span_sample_count=reference_detector.width,
            evaluated_candidate_bet_count=reference_detector.active_hypothesis_count,
        )
        assert detector_state_snapshot.last_observation == reference_observation


@pytest.mark.parametrize("class_count", [2, 3, 10])
@pytest.mark.parametrize("maximum_retained_candidate_count", [1, 7, 1000])
def test_loss_monitoring_overall_and_class_series_match_reference(
    valid_run_settings_mapping, monkeypatch, class_count, maximum_retained_candidate_count
):
    """class局所上昇・未観測class・切捨てを含む420観測の全状態を照合する。"""
    loss_change_detection_settings = valid_run_settings_mapping["loss_change_detection_settings"]
    monitor = OverallAndTrueClassLossMonitor(
        loss_change_detection_settings=loss_change_detection_settings,
        class_count=class_count,
        initial_baseline_loss_mean=0.6,
        maximum_retained_candidate_count=maximum_retained_candidate_count,
        betting_fractions=(0.05, 0.1, 0.2, 0.4, 0.8),
    )
    reference_monitor = make_reference_class_monitor(
        monkeypatch=monkeypatch,
        class_count=class_count,
        baseline_loss_mean=0.6,
        maximum_retained_candidate_count=maximum_retained_candidate_count,
        false_alarm_control_alpha=0.05,
    )
    assert_class_monitor_matches_reference(monitor=monitor, reference_monitor=reference_monitor)
    for observation_index in range(420):
        observed_class_id = observation_index % 2
        observed_loss = (
            (0.1 if observed_class_id == 0 else 0.5)
            if observation_index < 200
            else (1.0 if observed_class_id == 0 else 0.0)
        )
        reference_state = dict(reference_monitor.compute_counters)
        observation = monitor.observe_loss_after_label_observation(
            observed_loss=observed_loss,
            observed_class_id=observed_class_id,
            sample_index=observation_index + 100,
            current_model_baseline_loss_mean=0.6,
        )
        reference_detected = reference_monitor._update_drift_detectors(
            observed_loss, torch.tensor([[float(observed_class_id)]]), observation_index + 100
        )
        reference_span = reference_monitor._estimated_new_concept_span(observation_index + 100)
        assert observation.log_e_value == reference_monitor.history_detector_log_e[-1]
        assert observation.drift_detected == reference_detected
        assert observation.estimated_change_span_sample_count == reference_span
        assert (
            observation.detector_candidate_start_sample_index
            == observation_index + 100 - reference_span + 1
        )
        assert (
            observation.component_update_count
            == reference_monitor.compute_counters["drift_detector_updates"]
            - reference_state["drift_detector_updates"]
        )
        assert (
            observation.evaluated_candidate_bet_count
            == reference_monitor.compute_counters["drift_detector_hypotheses"]
            - reference_state["drift_detector_hypotheses"]
        )
        assert_class_monitor_matches_reference(monitor=monitor, reference_monitor=reference_monitor)


def test_loss_monitoring_first_class_observation_freezes_current_baseline(
    valid_run_settings_mapping, monkeypatch
):
    """class初出のbaselineだけを取り込み、既存系列を変更しない。"""
    monitor = OverallAndTrueClassLossMonitor(
        loss_change_detection_settings=valid_run_settings_mapping["loss_change_detection_settings"],
        class_count=3,
        initial_baseline_loss_mean=0.2,
        maximum_retained_candidate_count=7,
        betting_fractions=(0.05, 0.1, 0.2, 0.4, 0.8),
    )
    reference_monitor = make_reference_class_monitor(
        monkeypatch=monkeypatch,
        class_count=3,
        baseline_loss_mean=0.2,
        maximum_retained_candidate_count=7,
        false_alarm_control_alpha=0.05,
    )
    for sample_index, observed_class_id, current_model_baseline_loss_mean in (
        (0, 0, 0.2),
        (1, 1, 0.7),
        (2, 0, 0.4),
        (3, 2, 0.9),
        (4, 1, 0.1),
    ):
        reference_monitor.model_stats[0]["mean"] = current_model_baseline_loss_mean
        monitor.observe_loss_after_label_observation(
            observed_loss=0.3,
            observed_class_id=observed_class_id,
            sample_index=sample_index,
            current_model_baseline_loss_mean=current_model_baseline_loss_mean,
        )
        reference_monitor._update_drift_detectors(
            0.3, torch.tensor([[float(observed_class_id)]]), sample_index
        )
        assert_class_monitor_matches_reference(monitor=monitor, reference_monitor=reference_monitor)
    state_snapshot = monitor.get_state_snapshot()
    assert state_snapshot.overall_esr_state.baseline_loss_mean == 0.2
    assert tuple(
        reference_state.baseline_loss_mean
        for class_id, reference_state in state_snapshot.class_esr_states_by_class_id
    ) == (0.2, 0.7, 0.9)


def test_loss_monitoring_class_positions_survive_retention_and_reset(
    valid_run_settings_mapping, monkeypatch
):
    """class位置の保持/切捨てとreset後のglobal位置を旧経路へ照合する。"""
    monitor = OverallAndTrueClassLossMonitor(
        loss_change_detection_settings=valid_run_settings_mapping["loss_change_detection_settings"],
        class_count=2,
        initial_baseline_loss_mean=0.2,
        maximum_retained_candidate_count=2,
        betting_fractions=(0.05, 0.1, 0.2, 0.4, 0.8),
    )
    reference_monitor = make_reference_class_monitor(
        monkeypatch=monkeypatch,
        class_count=2,
        baseline_loss_mean=0.2,
        maximum_retained_candidate_count=2,
        false_alarm_control_alpha=0.05,
    )
    for sample_index in range(100, 120):
        observed_class_id = int(sample_index % 3 == 0)
        monitor.observe_loss_after_label_observation(
            observed_loss=0.8,
            observed_class_id=observed_class_id,
            sample_index=sample_index,
            current_model_baseline_loss_mean=0.2,
        )
        reference_monitor._update_drift_detectors(
            0.8, torch.tensor([[float(observed_class_id)]]), sample_index
        )
        assert_class_monitor_matches_reference(monitor=monitor, reference_monitor=reference_monitor)

    monitor.reset(baseline_loss_mean=0.7)
    reference_monitor.model_stats[0]["mean"] = 0.7
    reference_monitor._reset_drift_detectors()
    assert_class_monitor_matches_reference(monitor=monitor, reference_monitor=reference_monitor)
    assert monitor.last_observation is None
    assert monitor.get_state_snapshot().last_sample_index is None
    for sample_index in range(500, 505):
        observation = monitor.observe_loss_after_label_observation(
            observed_loss=0.8,
            observed_class_id=1,
            sample_index=sample_index,
            current_model_baseline_loss_mean=0.7,
        )
        reference_monitor._update_drift_detectors(0.8, torch.tensor([[1.0]]), sample_index)
        assert observation.log_e_value == reference_monitor.history_detector_log_e[-1]
        assert_class_monitor_matches_reference(monitor=monitor, reference_monitor=reference_monitor)


@pytest.mark.parametrize("best_component", ["overall", "class"])
def test_loss_monitoring_component_ties_preserve_reference_priority(
    valid_run_settings_mapping, monkeypatch, best_component
):
    """同じ寄与を注入し、overall→今回class→他classという優先順を分離検証する。"""
    monitor = OverallAndTrueClassLossMonitor(
        loss_change_detection_settings=valid_run_settings_mapping["loss_change_detection_settings"],
        class_count=2,
        initial_baseline_loss_mean=0.2,
        maximum_retained_candidate_count=7,
        betting_fractions=(0.05, 0.1, 0.2, 0.4, 0.8),
    )
    reference_monitor = make_reference_class_monitor(
        monkeypatch=monkeypatch,
        class_count=2,
        baseline_loss_mean=0.2,
        maximum_retained_candidate_count=7,
        false_alarm_control_alpha=0.05,
    )
    for sample_index, observed_class_id in ((100, 0), (101, 0), (102, 1)):
        monitor.observe_loss_after_label_observation(
            observed_loss=0.2,
            observed_class_id=observed_class_id,
            sample_index=sample_index,
            current_model_baseline_loss_mean=0.2,
        )
        reference_monitor._update_drift_detectors(
            0.2, torch.tensor([[float(observed_class_id)]]), sample_index
        )
    for class_id, detector, reference_detector in (
        (None, monitor._overall_detector, reference_monitor.e_detector),
        *(
            (
                class_id,
                monitor._class_detectors_by_class_id[class_id],
                reference_monitor.class_e_detectors[class_id],
            )
            for class_id in (0, 1)
        ),
    ):
        log_e_value = (0.0 if class_id is None and best_component == "class" else 100.0) - math.log(
            reference_monitor.overall_component_weight
            if class_id is None
            else reference_monitor.class_component_weight
        )
        monkeypatch.setattr(
            detector,
            "_last_observation",
            replace(detector.last_observation, log_e_value=log_e_value),
        )
        monkeypatch.setattr(
            detector,
            "observe_loss",
            lambda *, observed_loss, detector=detector: detector.last_observation,
        )
        monkeypatch.setattr(reference_detector, "log_e_value", log_e_value)
        monkeypatch.setattr(reference_detector, "update", lambda observed_loss: None)
    observation = monitor.observe_loss_after_label_observation(
        observed_loss=0.2,
        observed_class_id=1,
        sample_index=103,
        current_model_baseline_loss_mean=0.2,
    )
    reference_detected = reference_monitor._update_drift_detectors(0.2, torch.tensor([[1.0]]), 103)
    reference_span = reference_monitor._estimated_new_concept_span(103)
    assert reference_detected and observation.drift_detected
    assert observation.log_e_value == reference_monitor.history_detector_log_e[-1]
    assert observation.detector_candidate_start_sample_index == 103 - reference_span + 1
    assert observation.detector_candidate_start_sample_index == (
        102 if best_component == "class" else 101
    )


@pytest.mark.parametrize(
    "invalid_parameter_name,invalid_parameter_value",
    [
        ("observed_loss", True),
        ("observed_loss", float("nan")),
        ("observed_loss", -0.1),
        ("observed_class_id", True),
        ("observed_class_id", -1),
        ("observed_class_id", 2),
        ("observed_class_id", 1.0),
        ("sample_index", True),
        ("sample_index", -1),
        ("sample_index", 0),
        ("sample_index", 2),
        ("sample_index", 1.0),
        ("current_model_baseline_loss_mean", True),
        ("current_model_baseline_loss_mean", float("nan")),
        ("current_model_baseline_loss_mean", 1.1),
    ],
)
def test_loss_monitoring_invalid_inputs_preserve_complete_state_for_mixture(
    valid_run_settings_mapping, invalid_parameter_name, invalid_parameter_value
):
    """全体の更新前にclass・位置・baselineも検査し、両成分を保持する。"""
    monitor = OverallAndTrueClassLossMonitor(
        loss_change_detection_settings=valid_run_settings_mapping["loss_change_detection_settings"],
        class_count=2,
        initial_baseline_loss_mean=0.2,
        maximum_retained_candidate_count=2,
        betting_fractions=(0.1,),
    )
    monitor.observe_loss_after_label_observation(
        observed_loss=0.2, observed_class_id=0, sample_index=0, current_model_baseline_loss_mean=0.2
    )
    state_before_call = monitor.get_state_snapshot()
    with pytest.raises((TypeError, ValueError)) as exception_info:
        monitor.observe_loss_after_label_observation(
            **(
                dict(
                    observed_loss=0.2,
                    observed_class_id=1,
                    sample_index=1,
                    current_model_baseline_loss_mean=0.7,
                )
                | {invalid_parameter_name: invalid_parameter_value}
            )
        )
    assert str(exception_info.value)
    assert monitor.get_state_snapshot() == state_before_call

    with pytest.raises(ValueError):
        monitor.reset(baseline_loss_mean=float("nan"))
    assert monitor.get_state_snapshot() == state_before_call


def test_loss_monitoring_snapshots_and_instances_are_independent(valid_run_settings_mapping):
    """取得済みcapital/位置・immutable結果・他実体へ更新を波及させない。"""
    monitor_constructor_arguments = dict(
        loss_change_detection_settings=valid_run_settings_mapping["loss_change_detection_settings"],
        class_count=2,
        initial_baseline_loss_mean=0.2,
        maximum_retained_candidate_count=3,
        betting_fractions=(0.1,),
    )
    monitor = OverallAndTrueClassLossMonitor(**monitor_constructor_arguments)
    reference_monitor = OverallAndTrueClassLossMonitor(**monitor_constructor_arguments)
    reference_state = reference_monitor.get_state_snapshot()
    monitor.observe_loss_after_label_observation(
        observed_loss=0.8, observed_class_id=1, sample_index=0, current_model_baseline_loss_mean=0.2
    )
    state_snapshot = monitor.get_state_snapshot()
    with pytest.raises(FrozenInstanceError):
        state_snapshot.last_sample_index = 9
    with pytest.raises(FrozenInstanceError):
        monitor.last_observation.drift_detected = True
    with pytest.raises(TypeError):
        state_snapshot.class_sample_indices_by_class_id[0][1][0] = 9
    monitor.observe_loss_after_label_observation(
        observed_loss=0.2, observed_class_id=1, sample_index=1, current_model_baseline_loss_mean=0.2
    )
    assert state_snapshot.last_sample_index == 0
    assert state_snapshot.overall_esr_state.last_observation.observed_loss_count == 1
    assert state_snapshot.class_sample_indices_by_class_id == ((1, (0,)),)
    assert reference_monitor.get_state_snapshot() == reference_state
    monitor.reset(baseline_loss_mean=0.6)
    assert state_snapshot.overall_esr_state.baseline_loss_mean == 0.2
    assert reference_monitor.get_state_snapshot() == reference_state


@pytest.mark.parametrize("class_count", [2, 10])
def test_loss_monitoring_current_model_observed_loss_connects_without_prediction_state(
    valid_run_settings_mapping, monkeypatch, class_count
):
    """指定現行モデルの観測後損失のみを渡して旧ClassESRへ照合する。"""
    monitor = OverallAndTrueClassLossMonitor(
        loss_change_detection_settings=valid_run_settings_mapping["loss_change_detection_settings"],
        class_count=class_count,
        initial_baseline_loss_mean=0.6,
        maximum_retained_candidate_count=7,
        betting_fractions=(0.05, 0.1, 0.2, 0.4, 0.8),
    )
    reference_monitor = make_reference_class_monitor(
        monkeypatch=monkeypatch,
        class_count=class_count,
        baseline_loss_mean=0.6,
        maximum_retained_candidate_count=7,
        false_alarm_control_alpha=0.05,
    )
    for observation_index in range(20):
        model_outputs_by_model_id = {
            7: torch.tensor([[0.8]])
            if class_count == 2
            else torch.arange(class_count, dtype=torch.float32).reshape(1, class_count),
            -3: torch.tensor([[0.2]])
            if class_count == 2
            else -torch.arange(class_count, dtype=torch.float32).reshape(1, class_count),
        }
        prediction_probabilities_by_model_id = convert_model_outputs_to_prediction_probabilities(
            model_outputs_by_model_id=model_outputs_by_model_id, class_count=class_count
        )
        observed_class_id = observation_index % class_count
        observed_class_labels = torch.tensor([observed_class_id])
        observed_losses_by_model_id = compute_model_mean_bounded_losses_after_label_observation(
            prediction_probabilities_by_model_id=prediction_probabilities_by_model_id,
            observed_class_labels=observed_class_labels,
            class_count=class_count,
        )
        current_training_model_id = 7 if observation_index < 10 else -3
        observed_loss = _AdaHedgeRoutingFedSDAClientMixin._routing_score_loss(
            prediction_probabilities_by_model_id[current_training_model_id],
            observed_class_labels,
            class_count,
        )
        assert observed_losses_by_model_id[current_training_model_id] == observed_loss
        observation = monitor.observe_loss_after_label_observation(
            observed_loss=observed_losses_by_model_id[current_training_model_id],
            observed_class_id=observed_class_id,
            sample_index=observation_index,
            current_model_baseline_loss_mean=0.6,
        )
        reference_detected = reference_monitor._update_drift_detectors(
            observed_loss, observed_class_labels, observation_index
        )
        assert observation.drift_detected == reference_detected
        assert observation.log_e_value == reference_monitor.history_detector_log_e[-1]
        assert_class_monitor_matches_reference(monitor=monitor, reference_monitor=reference_monitor)


def test_loss_monitoring_public_calls_preserve_random_states(valid_run_settings_mapping):
    """constructor・更新・copy・resetで共有乱数を消費しない。"""
    global_python_random_state = random.getstate()
    global_numpy_random_state = np.random.get_state()
    global_torch_random_state = torch.random.get_rng_state().clone()
    monitor = OverallAndTrueClassLossMonitor(
        loss_change_detection_settings=valid_run_settings_mapping["loss_change_detection_settings"],
        class_count=2,
        initial_baseline_loss_mean=0.2,
        maximum_retained_candidate_count=7,
        betting_fractions=(0.1, 0.4),
    )
    for sample_index in range(3):
        monitor.observe_loss_after_label_observation(
            observed_loss=0.8,
            observed_class_id=sample_index % 2,
            sample_index=sample_index,
            current_model_baseline_loss_mean=0.2,
        )
        monitor.get_state_snapshot()
    monitor.reset(baseline_loss_mean=0.6)
    assert random.getstate() == global_python_random_state
    assert np.random.get_state()[0] == global_numpy_random_state[0]
    assert np.array_equal(np.random.get_state()[1], global_numpy_random_state[1])
    assert np.random.get_state()[2:] == global_numpy_random_state[2:]
    assert torch.equal(torch.random.get_rng_state(), global_torch_random_state)


@pytest.mark.parametrize(
    "operation",
    [
        BoundedLossESRDetector,
        BoundedLossESRDetector.observe_loss,
        BoundedLossESRDetector.reset,
        OverallAndTrueClassLossMonitor,
        OverallAndTrueClassLossMonitor.observe_loss_after_label_observation,
        OverallAndTrueClassLossMonitor.reset,
    ],
)
def test_loss_monitoring_public_functions_require_explicit_keyword_arguments(operation):
    """位置・class・baselineをkeywordで明示し、Tensor/モデルを受け取らない。"""
    assert "*, " in str(inspect.signature(operation))
    with pytest.raises(TypeError):
        operation(object(), object())
