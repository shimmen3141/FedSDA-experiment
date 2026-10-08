"""警報応答の完了処理（監視の再開・保留位置の消費・記録用情報）を実旧の警報処理と照合する。"""

import random
from collections import defaultdict, deque
from dataclasses import FrozenInstanceError, asdict, replace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from test_alarm_buffer_response import (
    assert_buffer_response_matches_legacy,
    build_buffer_response_oracle,
)
from test_loss_change_monitoring import assert_class_monitor_matches_reference
from test_run_settings_validation import valid_run_settings_mapping as valid_run_settings_mapping

from federated_drift_experiment import config
from federated_drift_experiment.adaptation_events import AdaptationEvent
from federated_drift_experiment.drift_detectors.e_detector import BoundedMeanEDetector
from federated_learning_experiments.learning.training.indexed_observed_training_sample import (
    IndexedObservedTrainingSample,
)
from federated_learning_experiments.methods.fedsda.loss_change_detection.overall_and_true_class_loss_monitoring import (
    OverallAndTrueClassLossMonitor,
)
from federated_learning_experiments.methods.fedsda.loss_statistics.loss_baseline_selection import (
    select_loss_monitoring_baseline_mean_loss,
)
from federated_learning_experiments.runtime.alarm_buffer_response import (
    respond_to_alarm_with_buffered_samples,
)
from federated_learning_experiments.runtime.alarm_response_completion import (
    AlarmResponseCompletion,
    complete_alarm_buffer_response,
)

LEGACY_ACTION_BY_RESPONSE_OUTCOME = {
    "alarm_during_candidate_validation": "forward_validation_pending",
    "alarm_change_interval_too_short": "insufficient_data",
    "alarm_interval_held_model_reused": "reuse",
    "alarm_interval_current_model_maintained": "maintain",
    "alarm_interval_candidate_validation_started": "create_pending",
}
LEGACY_REUSE_SELECTION_COUNTS_BY_RESPONSE_OUTCOME = {
    "alarm_during_candidate_validation": {},
    "alarm_change_interval_too_short": {},
    "alarm_interval_held_model_reused": {"alternative_fit": 1},
    "alarm_interval_current_model_maintained": {"current_fit": 1},
    "alarm_interval_candidate_validation_started": {},
}
LOSS_MONITOR_BETTING_FRACTIONS = (0.05, 0.1, 0.2, 0.4, 0.8)
MAXIMUM_RETAINED_CANDIDATE_COUNT = 7
MONITORED_LOSSES_BEFORE_ALARM = (0.1, 0.2, 0.1, 0.9, 1.0, 0.8, 1.0, 0.9, 1.0)
MONITORED_LOSSES_AFTER_COMPLETION = (0.3, 0.2, 0.9, 1.0, 1.0, 0.1, 1.0, 1.0)


def select_current_model_monitoring_baseline(*, completion_arguments):
    """新ownerの現行モデル統計から、公開の基準選択で監視の基準平均を得る。"""
    current_model_loss_statistics = completion_arguments[
        "loss_statistics_store"
    ].get_model_loss_statistics(
        model_id=completion_arguments["current_training_model_assignment"].current_training_model_id
    )
    return select_loss_monitoring_baseline_mean_loss(
        loss_moments=None
        if current_model_loss_statistics is None
        else current_model_loss_statistics.overall_loss_moments
    )


def observe_monitored_losses_in_both_implementations(
    *, completion_arguments, legacy_client, class_count, first_sample_index, monitored_losses
):
    """同じ損失列を新監視と実旧ClassESRへ与え、各観測後の全状態を照合する。"""
    loss_change_monitor = completion_arguments["loss_change_monitor"]
    for sample_offset, monitored_loss in enumerate(monitored_losses):
        sample_index = first_sample_index + sample_offset
        observed_class_id = sample_offset % class_count
        monitoring_observation = loss_change_monitor.observe_loss_after_label_observation(
            observed_loss=monitored_loss,
            observed_class_id=observed_class_id,
            sample_index=sample_index,
            current_model_baseline_loss_mean=select_current_model_monitoring_baseline(
                completion_arguments=completion_arguments
            ),
        )
        legacy_drift_detected = legacy_client._update_drift_detectors(
            monitored_loss, torch.tensor([[float(observed_class_id)]]), sample_index
        )
        assert monitoring_observation.log_e_value == legacy_client.history_detector_log_e[-1]
        assert monitoring_observation.drift_detected == legacy_drift_detected
        assert monitoring_observation.estimated_change_span_sample_count == type(
            legacy_client
        )._estimated_new_concept_span(legacy_client, sample_index)
        assert_class_monitor_matches_reference(
            monitor=loss_change_monitor, reference_monitor=legacy_client
        )


def build_response_completion_oracle(
    *,
    monkeypatch,
    loss_change_detection_settings,
    class_count=2,
    alarm_interval_resolution_case="current_model_maintained",
    earlier_sample_count=3,
    estimated_change_span_sample_count=3,
    minimum_change_interval_sample_count=3,
):
    """警報応答のoracleへ、同じ損失列を観測済みの新監視と実旧ClassESRの検出状態を加える。"""
    (
        response_arguments,
        preparation_arguments,
        resolution_arguments,
        shared_optimizer_owners,
        legacy_client,
    ) = build_buffer_response_oracle(
        monkeypatch=monkeypatch,
        class_count=class_count,
        alarm_interval_resolution_case=alarm_interval_resolution_case,
        earlier_sample_count=earlier_sample_count,
        estimated_change_span_sample_count=estimated_change_span_sample_count,
        minimum_change_interval_sample_count=minimum_change_interval_sample_count,
    )
    false_alarm_control_alpha = float(loss_change_detection_settings.e_sr_false_alarm_control_alpha)
    monkeypatch.setattr(config, "num_classes", lambda: class_count)
    monkeypatch.setattr(config, "E_DETECTOR_ALPHA", false_alarm_control_alpha)
    monkeypatch.setattr(config, "ADWIN_MAX_WINDOW", MAXIMUM_RETAINED_CANDIDATE_COUNT)
    # 実旧の基準選択・イベント記録・検出器resetを使うため、上流oracleの差し替えを外す。
    for replaced_method_name in ("_record_adaptation_event", "_reset_drift_detectors"):
        legacy_client.__dict__.pop(replaced_method_name, None)
    legacy_client.e_detector = BoundedMeanEDetector(
        legacy_client._e_detector_baseline(),
        false_alarm_control_alpha,
        MAXIMUM_RETAINED_CANDIDATE_COUNT,
    )
    legacy_client.overall_component_weight = 1.0 / (class_count + 1)
    legacy_client.class_component_weight = (
        1.0 - legacy_client.overall_component_weight
    ) / class_count
    legacy_client.class_e_detectors = {}
    legacy_client.class_e_positions = defaultdict(deque)
    legacy_client._class_drift_start = None
    legacy_client.history_detector_log_e = []
    legacy_client.compute_counters = defaultdict(int)
    legacy_client.adaptation_events = []
    legacy_client.local_switch_positions = []
    completion_arguments = dict(
        alarm_sample_index=response_arguments["proposal_sample_index"],
        estimated_change_point_sample_index=response_arguments[
            "estimated_change_point_sample_index"
        ],
        detection_episode_id=response_arguments["detection_episode_id"],
        current_training_model_assignment=response_arguments["current_training_model_assignment"],
        loss_statistics_store=response_arguments["loss_statistics_store"],
        pending_training_assignment_buffer=response_arguments["pending_training_assignment_buffer"],
    )
    completion_arguments["loss_change_monitor"] = OverallAndTrueClassLossMonitor(
        loss_change_detection_settings=loss_change_detection_settings,
        class_count=class_count,
        initial_baseline_loss_mean=select_current_model_monitoring_baseline(
            completion_arguments=completion_arguments
        ),
        maximum_retained_candidate_count=MAXIMUM_RETAINED_CANDIDATE_COUNT,
        betting_fractions=LOSS_MONITOR_BETTING_FRACTIONS,
    )
    observe_monitored_losses_in_both_implementations(
        completion_arguments=completion_arguments,
        legacy_client=legacy_client,
        class_count=class_count,
        first_sample_index=response_arguments["proposal_sample_index"]
        - len(MONITORED_LOSSES_BEFORE_ALARM)
        + 1,
        monitored_losses=MONITORED_LOSSES_BEFORE_ALARM,
    )
    return (
        response_arguments,
        completion_arguments,
        preparation_arguments,
        resolution_arguments,
        shared_optimizer_owners,
        legacy_client,
    )


def run_legacy_alarm_with_real_completion(*, response_arguments, legacy_client, monkeypatch):
    """実旧_resolve_driftを、イベント記録・検出器reset・FIFO clearも差し替えずに実行する。"""
    global_python_random_state = random.getstate()
    recorded_event_count = len(legacy_client.adaptation_events)
    legacy_client._estimated_new_concept_span = lambda sample_index: response_arguments[
        "estimated_change_span_sample_count"
    ]
    legacy_client.reuse_selection_counts = defaultdict(int)
    legacy_client.local_switch_positions = []
    legacy_client.buffer = deque(
        (
            indexed_observation.training_sample.input_features,
            indexed_observation.training_sample.observed_class_labels,
            indexed_observation.observed_concept_id,
        )
        for indexed_observation in response_arguments["pending_sample_observations"]
    )
    monkeypatch.setattr(
        config, "MIN_DRIFT_DATA", response_arguments["minimum_change_interval_sample_count"]
    )
    try:
        random.setstate(response_arguments["python_random_generator"].getstate())
        with patch.object(
            legacy_client,
            "_update_new_model_epochs",
            wraps=legacy_client._update_new_model_epochs,
        ) as legacy_epoch_training_calls:
            legacy_drift_type = legacy_client._resolve_drift(
                response_arguments["proposal_sample_index"],
                response_arguments["estimated_change_point_sample_index"],
                response_arguments["detection_episode_id"],
            )
        legacy_python_random_state = random.getstate()
    finally:
        random.setstate(global_python_random_state)
    legacy_adaptation_events = legacy_client.adaptation_events[recorded_event_count:]
    assert len(legacy_adaptation_events) == 1
    assert type(legacy_adaptation_events[0]) is AdaptationEvent
    # 警報応答の照合helperはイベントをfieldのdictで受け取る。
    return (
        legacy_drift_type,
        [asdict(legacy_adaptation_events[0])],
        legacy_epoch_training_calls,
        legacy_python_random_state,
    )


def assert_response_completion_matches_legacy(
    *,
    response_completion,
    completion_arguments,
    legacy_client,
    legacy_result,
    pending_sample_indices_before_completion,
):
    """記録用の情報・保留位置・監視状態を、実旧のイベント・FIFO・検出器と照合する。"""
    legacy_drift_type, legacy_adaptation_events, _, _ = legacy_result
    legacy_adaptation_event = legacy_adaptation_events[0]
    response_outcome = response_completion.alarm_buffer_response.response_outcome
    assert type(response_completion) is AlarmResponseCompletion
    assert legacy_adaptation_event == dict(
        position=response_completion.alarm_sample_index,
        detector=legacy_client._detector_label(),
        action=LEGACY_ACTION_BY_RESPONSE_OUTCOME[response_outcome],
        old_model_id=response_completion.previous_training_model_id,
        new_model_id=response_completion.current_training_model_id,
        estimated_change_point=response_completion.estimated_change_point_sample_index,
        episode_id=response_completion.detection_episode_id,
    )
    assert response_completion.current_training_model_id == legacy_client.current_model_id
    assert response_completion.detection_episode_operation_required == (legacy_drift_type in (1, 2))
    assert legacy_client.local_switch_positions == (
        []
        if response_completion.training_model_switch_sample_index is None
        else [response_completion.training_model_switch_sample_index]
    )
    assert (
        dict(legacy_client.reuse_selection_counts)
        == LEGACY_REUSE_SELECTION_COUNTS_BY_RESPONSE_OUTCOME[response_outcome]
    )
    pending_assignment_state = completion_arguments[
        "pending_training_assignment_buffer"
    ].get_state_snapshot()
    assert len(pending_assignment_state.pending_sample_indices) == len(legacy_client.buffer)
    if response_outcome == "alarm_change_interval_too_short":
        assert response_completion.drained_pending_sample_indices == ()
        assert (
            pending_assignment_state.pending_sample_indices
            == pending_sample_indices_before_completion
        )
    else:
        assert (
            response_completion.drained_pending_sample_indices
            == pending_sample_indices_before_completion
        )
        assert pending_assignment_state.pending_sample_indices == ()
    assert pending_assignment_state.last_observed_sample_index in (
        None,
        response_completion.alarm_sample_index,
    )
    loss_change_monitor = completion_arguments["loss_change_monitor"]
    assert (
        response_completion.loss_monitoring_baseline_mean_loss == legacy_client.e_detector.baseline
    )
    assert loss_change_monitor.last_observation is None
    assert legacy_client._class_drift_start is None
    assert_class_monitor_matches_reference(
        monitor=loss_change_monitor, reference_monitor=legacy_client
    )


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize(
    "alarm_interval_resolution_case",
    ["other_model_reused", "current_model_maintained", "no_model_fits"],
)
@pytest.mark.parametrize(
    "earlier_sample_count,estimated_change_span_sample_count,minimum_change_interval_sample_count",
    [(3, 3, 3), (3, 2, 3), (0, 99, 3), (3, 6, 7)],
)
def test_response_completion_matches_real_legacy(
    class_count,
    alarm_interval_resolution_case,
    earlier_sample_count,
    estimated_change_span_sample_count,
    minimum_change_interval_sample_count,
    monkeypatch,
    valid_run_settings_mapping,
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(823)
        (
            response_arguments,
            completion_arguments,
            preparation_arguments,
            resolution_arguments,
            shared_optimizer_owners,
            legacy_client,
        ) = build_response_completion_oracle(
            monkeypatch=monkeypatch,
            loss_change_detection_settings=valid_run_settings_mapping[
                "loss_change_detection_settings"
            ],
            class_count=class_count,
            alarm_interval_resolution_case=alarm_interval_resolution_case,
            earlier_sample_count=earlier_sample_count,
            estimated_change_span_sample_count=estimated_change_span_sample_count,
            minimum_change_interval_sample_count=minimum_change_interval_sample_count,
        )
        initial_training_model_id = legacy_client.current_model_id
        initial_torch_random_state = torch.get_rng_state().clone()
        legacy_result = run_legacy_alarm_with_real_completion(
            response_arguments=response_arguments,
            legacy_client=legacy_client,
            monkeypatch=monkeypatch,
        )
        expected_torch_random_state = torch.get_rng_state().clone()
        torch.set_rng_state(initial_torch_random_state)
        alarm_buffer_response = respond_to_alarm_with_buffered_samples(**response_arguments)
        pending_sample_indices_before_completion = (
            completion_arguments["pending_training_assignment_buffer"]
            .get_state_snapshot()
            .pending_sample_indices
        )
        python_random_state = response_arguments["python_random_generator"].getstate()
        global_python_random_state = random.getstate()
        numpy_random_state = np.random.get_state()
        torch_random_state_before_completion = torch.get_rng_state().clone()
        response_completion = complete_alarm_buffer_response(
            alarm_buffer_response=alarm_buffer_response, **completion_arguments
        )
        # 完了処理は読取りと監視・保留位置の更新だけで、どの乱数も消費しない。
        assert torch.equal(torch.get_rng_state(), torch_random_state_before_completion)
        assert torch.equal(torch.get_rng_state(), expected_torch_random_state)
        assert response_arguments["python_random_generator"].getstate() == python_random_state
        assert random.getstate() == global_python_random_state
        assert np.array_equal(np.random.get_state()[1], numpy_random_state[1])
        assert response_completion.alarm_buffer_response is alarm_buffer_response
        assert_response_completion_matches_legacy(
            response_completion=response_completion,
            completion_arguments=completion_arguments,
            legacy_client=legacy_client,
            legacy_result=legacy_result,
            pending_sample_indices_before_completion=pending_sample_indices_before_completion,
        )
        assert_buffer_response_matches_legacy(
            response=alarm_buffer_response,
            response_arguments=response_arguments,
            preparation_arguments=preparation_arguments,
            resolution_arguments=resolution_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
            legacy_client=legacy_client,
            legacy_result=legacy_result,
            initial_training_model_id=initial_training_model_id,
        )
        # 再開した監視は、切替後の現行モデルの統計を基準に実旧と同じ値で進む。
        observe_monitored_losses_in_both_implementations(
            completion_arguments=completion_arguments,
            legacy_client=legacy_client,
            class_count=class_count,
            first_sample_index=response_arguments["proposal_sample_index"] + 1,
            monitored_losses=MONITORED_LOSSES_AFTER_COMPLETION,
        )


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize("second_alarm_sample_count", [0, 3])
def test_second_alarm_during_started_validation_completes_like_legacy(
    class_count, second_alarm_sample_count, monkeypatch, valid_run_settings_mapping
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(823)
        (
            response_arguments,
            completion_arguments,
            preparation_arguments,
            resolution_arguments,
            shared_optimizer_owners,
            legacy_client,
        ) = build_response_completion_oracle(
            monkeypatch=monkeypatch,
            loss_change_detection_settings=valid_run_settings_mapping[
                "loss_change_detection_settings"
            ],
            class_count=class_count,
            alarm_interval_resolution_case="no_model_fits",
            earlier_sample_count=0,
        )
        initial_training_model_id = legacy_client.current_model_id
        initial_torch_random_state = torch.get_rng_state().clone()
        run_legacy_alarm_with_real_completion(
            response_arguments=response_arguments,
            legacy_client=legacy_client,
            monkeypatch=monkeypatch,
        )
        torch.set_rng_state(initial_torch_random_state)
        first_response_completion = complete_alarm_buffer_response(
            alarm_buffer_response=respond_to_alarm_with_buffered_samples(**response_arguments),
            **completion_arguments,
        )
        started_validation_session = (
            first_response_completion.alarm_buffer_response.active_validation_session
        )
        legacy_session = legacy_client._forward_validation
        assert started_validation_session is not None and legacy_session is not None
        # 候補検証の開始後に届いた標本を保留し、同じ損失を監視へ与えてから2回目の警報とする。
        first_alarm_observations = response_arguments["pending_sample_observations"]
        second_alarm_observations = tuple(
            IndexedObservedTrainingSample(
                sample_index=response_arguments["proposal_sample_index"] + 1 + sample_offset,
                training_sample=first_alarm_observations[sample_offset].training_sample,
                observed_concept_id=first_alarm_observations[sample_offset].observed_concept_id,
            )
            for sample_offset in range(second_alarm_sample_count)
        )
        for indexed_observation in second_alarm_observations:
            completion_arguments["pending_training_assignment_buffer"].append_observed_sample_index(
                sample_index=indexed_observation.sample_index
            )
        observe_monitored_losses_in_both_implementations(
            completion_arguments=completion_arguments,
            legacy_client=legacy_client,
            class_count=class_count,
            first_sample_index=response_arguments["proposal_sample_index"] + 1,
            monitored_losses=MONITORED_LOSSES_AFTER_COMPLETION[:second_alarm_sample_count],
        )
        second_alarm_sample_index = (
            response_arguments["proposal_sample_index"] + second_alarm_sample_count
        )
        response_arguments.update(
            active_validation_session=started_validation_session,
            pending_sample_observations=second_alarm_observations,
            proposal_sample_index=second_alarm_sample_index,
            estimated_change_point_sample_index=second_alarm_sample_index,
        )
        completion_arguments.update(
            alarm_sample_index=second_alarm_sample_index,
            estimated_change_point_sample_index=second_alarm_sample_index,
        )
        torch_random_state_before_second_alarm = torch.get_rng_state().clone()
        legacy_result = run_legacy_alarm_with_real_completion(
            response_arguments=response_arguments,
            legacy_client=legacy_client,
            monkeypatch=monkeypatch,
        )
        assert torch.equal(torch.get_rng_state(), torch_random_state_before_second_alarm)
        second_alarm_response = respond_to_alarm_with_buffered_samples(**response_arguments)
        assert torch.equal(torch.get_rng_state(), torch_random_state_before_second_alarm)
        second_response_completion = complete_alarm_buffer_response(
            alarm_buffer_response=second_alarm_response, **completion_arguments
        )
        assert (
            second_alarm_response.response_outcome == "alarm_during_candidate_validation"
            and second_alarm_response.active_validation_session is started_validation_session
            and legacy_client._forward_validation is legacy_session
        )
        assert second_response_completion.previous_training_model_id == initial_training_model_id
        assert_response_completion_matches_legacy(
            response_completion=second_response_completion,
            completion_arguments=completion_arguments,
            legacy_client=legacy_client,
            legacy_result=legacy_result,
            pending_sample_indices_before_completion=tuple(
                indexed_observation.sample_index
                for indexed_observation in second_alarm_observations
            ),
        )
        assert_buffer_response_matches_legacy(
            response=second_alarm_response,
            response_arguments=response_arguments,
            preparation_arguments=preparation_arguments,
            resolution_arguments=resolution_arguments,
            shared_optimizer_owners=shared_optimizer_owners,
            legacy_client=legacy_client,
            legacy_result=legacy_result,
            initial_training_model_id=initial_training_model_id,
        )
        observe_monitored_losses_in_both_implementations(
            completion_arguments=completion_arguments,
            legacy_client=legacy_client,
            class_count=class_count,
            first_sample_index=second_alarm_sample_index + 1,
            monitored_losses=MONITORED_LOSSES_AFTER_COMPLETION,
        )


@pytest.mark.parametrize(
    "legacy_model_statistics,expected_baseline_mean_loss",
    [
        (None, 0.01),
        ({"n": 0, "mean": 0.0, "M2": 0.0}, 0.01),
        ({"n": 1, "mean": 0.001, "M2": 0.0}, 0.01),
        ({"n": 5, "mean": 0.5, "M2": 0.0}, 0.5),
        ({"n": 5, "mean": 1.0, "M2": 0.0}, 1.0 - 1e-6),
    ],
)
def test_completion_baseline_follows_current_model_statistics_like_legacy(
    legacy_model_statistics, expected_baseline_mean_loss, monkeypatch, valid_run_settings_mapping
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(823)
        (
            response_arguments,
            completion_arguments,
            _,
            _,
            _,
            legacy_client,
        ) = build_response_completion_oracle(
            monkeypatch=monkeypatch,
            loss_change_detection_settings=valid_run_settings_mapping[
                "loss_change_detection_settings"
            ],
            estimated_change_span_sample_count=6,
            minimum_change_interval_sample_count=99,
            earlier_sample_count=0,
        )
        # 変化区間が不足する応答は統計を更新しないので、応答後に現行モデルの統計だけを置き換える。
        alarm_buffer_response = respond_to_alarm_with_buffered_samples(**response_arguments)
        assert alarm_buffer_response.response_outcome == "alarm_change_interval_too_short"
        loss_statistics_store = completion_arguments["loss_statistics_store"]
        current_training_model_id = legacy_client.current_model_id
        if legacy_model_statistics is None:
            del legacy_client.model_stats[current_training_model_id]
            loss_statistics_store.reassign_model_loss_statistics_id(
                original_model_id=current_training_model_id, reassigned_model_id=-999
            )
        else:
            legacy_client.model_stats[current_training_model_id] = dict(legacy_model_statistics)
            loss_statistics_store.set_model_loss_statistics(
                model_id=current_training_model_id,
                loss_statistics=replace(
                    loss_statistics_store.get_model_loss_statistics(
                        model_id=current_training_model_id
                    ),
                    overall_loss_moments=replace(
                        loss_statistics_store.get_model_loss_statistics(
                            model_id=current_training_model_id
                        ).overall_loss_moments,
                        observed_loss_count=legacy_model_statistics["n"],
                        mean_loss=legacy_model_statistics["mean"],
                        sum_squared_loss_deviations=legacy_model_statistics["M2"],
                    ),
                ),
            )
        type(legacy_client)._reset_drift_detectors(legacy_client)
        response_completion = complete_alarm_buffer_response(
            alarm_buffer_response=alarm_buffer_response, **completion_arguments
        )
        assert response_completion.loss_monitoring_baseline_mean_loss == expected_baseline_mean_loss
        assert legacy_client.e_detector.baseline == expected_baseline_mean_loss
        assert response_completion.training_model_switch_sample_index is None
        assert response_completion.detection_episode_operation_required is False
        assert response_completion.drained_pending_sample_indices == ()
        # 不足の応答は保留位置を保持するので、消費済みの位置を持つ記録は作れない。
        with pytest.raises(ValueError):
            replace(response_completion, drained_pending_sample_indices=(0,))
        with pytest.raises(ValueError):
            replace(response_completion, previous_training_model_id=-999)
        assert_class_monitor_matches_reference(
            monitor=completion_arguments["loss_change_monitor"], reference_monitor=legacy_client
        )


def assign_other_model_after_response(*, completion_arguments, legacy_client):
    completion_arguments["current_training_model_assignment"].assign_model_for_training(
        model_id=next(
            model_id
            for model_id in legacy_client.models
            if model_id
            != completion_arguments["current_training_model_assignment"].current_training_model_id
        )
    )


def append_sample_index_after_response(*, completion_arguments, legacy_client):
    completion_arguments["pending_training_assignment_buffer"].append_observed_sample_index(
        sample_index=completion_arguments["alarm_sample_index"] + 1
    )
    completion_arguments["alarm_sample_index"] += 1


INVALID_COMPLETION_INPUT_CASES = {
    "response_is_not_exact_record": (
        lambda completion_arguments, legacy_client: completion_arguments.update(
            alarm_buffer_response=object()
        ),
        TypeError,
    ),
    "assignment_is_not_exact_owner": (
        lambda completion_arguments, legacy_client: completion_arguments.update(
            current_training_model_assignment=object()
        ),
        TypeError,
    ),
    "statistics_store_is_not_exact_owner": (
        lambda completion_arguments, legacy_client: completion_arguments.update(
            loss_statistics_store=object()
        ),
        TypeError,
    ),
    "monitor_is_not_exact_owner": (
        lambda completion_arguments, legacy_client: completion_arguments.update(
            loss_change_monitor=object()
        ),
        TypeError,
    ),
    "buffer_is_not_exact_owner": (
        lambda completion_arguments, legacy_client: completion_arguments.update(
            pending_training_assignment_buffer=object()
        ),
        TypeError,
    ),
    "alarm_index_is_bool": (
        lambda completion_arguments, legacy_client: completion_arguments.update(
            alarm_sample_index=True
        ),
        TypeError,
    ),
    "alarm_index_is_float": (
        lambda completion_arguments, legacy_client: completion_arguments.update(
            alarm_sample_index=float(completion_arguments["alarm_sample_index"])
        ),
        TypeError,
    ),
    "alarm_index_is_negative": (
        lambda completion_arguments, legacy_client: completion_arguments.update(
            alarm_sample_index=-1
        ),
        ValueError,
    ),
    "alarm_index_is_not_last_observed": (
        lambda completion_arguments, legacy_client: completion_arguments.update(
            alarm_sample_index=completion_arguments["alarm_sample_index"] + 1
        ),
        ValueError,
    ),
    "change_point_is_bool": (
        lambda completion_arguments, legacy_client: completion_arguments.update(
            estimated_change_point_sample_index=False
        ),
        TypeError,
    ),
    "change_point_is_negative": (
        lambda completion_arguments, legacy_client: completion_arguments.update(
            estimated_change_point_sample_index=-1
        ),
        ValueError,
    ),
    "episode_is_bool": (
        lambda completion_arguments, legacy_client: completion_arguments.update(
            detection_episode_id=True
        ),
        TypeError,
    ),
    "episode_is_negative": (
        lambda completion_arguments, legacy_client: completion_arguments.update(
            detection_episode_id=-1
        ),
        ValueError,
    ),
    "assignment_changed_after_response": (assign_other_model_after_response, ValueError),
    "pending_indices_changed_after_response": (append_sample_index_after_response, ValueError),
}


@pytest.mark.parametrize("invalid_case", INVALID_COMPLETION_INPUT_CASES)
def test_completion_rejects_invalid_inputs_before_updates(
    invalid_case, monkeypatch, valid_run_settings_mapping
):
    apply_invalid_input, expected_exception = INVALID_COMPLETION_INPUT_CASES[invalid_case]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(823)
        response_arguments, completion_arguments, _, _, _, legacy_client = (
            build_response_completion_oracle(
                monkeypatch=monkeypatch,
                loss_change_detection_settings=valid_run_settings_mapping[
                    "loss_change_detection_settings"
                ],
                alarm_interval_resolution_case="other_model_reused",
                earlier_sample_count=0,
            )
        )
        completion_arguments["alarm_buffer_response"] = respond_to_alarm_with_buffered_samples(
            **response_arguments
        )
        assert (
            completion_arguments["alarm_buffer_response"].response_outcome
            == "alarm_interval_held_model_reused"
        )
        valid_completion_arguments = dict(completion_arguments)
        apply_invalid_input(completion_arguments=completion_arguments, legacy_client=legacy_client)
        monitoring_state = valid_completion_arguments["loss_change_monitor"].get_state_snapshot()
        pending_assignment_state = valid_completion_arguments[
            "pending_training_assignment_buffer"
        ].get_state_snapshot()
        loss_statistics_state = valid_completion_arguments[
            "loss_statistics_store"
        ].get_state_snapshot()
        current_training_model_id = valid_completion_arguments[
            "current_training_model_assignment"
        ].current_training_model_id
        torch_random_state = torch.get_rng_state().clone()
        global_python_random_state = random.getstate()
        numpy_random_state = np.random.get_state()
        with pytest.raises(expected_exception):
            complete_alarm_buffer_response(**completion_arguments)
        assert (
            valid_completion_arguments["loss_change_monitor"].get_state_snapshot()
            == monitoring_state
        )
        assert (
            valid_completion_arguments["pending_training_assignment_buffer"].get_state_snapshot()
            == pending_assignment_state
        )
        assert (
            valid_completion_arguments["loss_statistics_store"].get_state_snapshot()
            == loss_statistics_state
        )
        assert (
            valid_completion_arguments[
                "current_training_model_assignment"
            ].current_training_model_id
            == current_training_model_id
        )
        assert torch.equal(torch.get_rng_state(), torch_random_state)
        assert random.getstate() == global_python_random_state
        assert np.array_equal(np.random.get_state()[1], numpy_random_state[1])


def test_repeated_completion_of_consumed_response_is_rejected(
    monkeypatch, valid_run_settings_mapping
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(823)
        response_arguments, completion_arguments, _, _, _, _ = build_response_completion_oracle(
            monkeypatch=monkeypatch,
            loss_change_detection_settings=valid_run_settings_mapping[
                "loss_change_detection_settings"
            ],
        )
        alarm_buffer_response = respond_to_alarm_with_buffered_samples(**response_arguments)
        complete_alarm_buffer_response(
            alarm_buffer_response=alarm_buffer_response, **completion_arguments
        )
        monitoring_state = completion_arguments["loss_change_monitor"].get_state_snapshot()
        # 保留位置を消費済みの応答は、準備済み区間と保留位置が一致しないので再適用できない。
        with pytest.raises(ValueError):
            complete_alarm_buffer_response(
                alarm_buffer_response=alarm_buffer_response, **completion_arguments
            )
        assert completion_arguments["loss_change_monitor"].get_state_snapshot() == monitoring_state


def test_completion_resets_monitor_with_selected_baseline_before_draining(
    monkeypatch, valid_run_settings_mapping
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(823)
        response_arguments, completion_arguments, _, _, _, _ = build_response_completion_oracle(
            monkeypatch=monkeypatch,
            loss_change_detection_settings=valid_run_settings_mapping[
                "loss_change_detection_settings"
            ],
            alarm_interval_resolution_case="other_model_reused",
            earlier_sample_count=0,
        )
        alarm_buffer_response = respond_to_alarm_with_buffered_samples(**response_arguments)
        assert alarm_buffer_response.response_outcome == "alarm_interval_held_model_reused"
        # 基準平均は、切替後の現行モデルが変化区間を吸収した後の統計から選ばれる。
        expected_baseline_mean_loss = select_current_model_monitoring_baseline(
            completion_arguments=completion_arguments
        )
        completion_operation_calls = []
        loss_change_monitor = completion_arguments["loss_change_monitor"]
        pending_training_assignment_buffer = completion_arguments[
            "pending_training_assignment_buffer"
        ]
        reset_loss_change_monitor = loss_change_monitor.reset
        drain_pending_sample_indices = (
            pending_training_assignment_buffer.drain_pending_sample_indices
        )

        def record_monitor_reset(*, baseline_loss_mean):
            completion_operation_calls.append(("reset", baseline_loss_mean))
            return reset_loss_change_monitor(baseline_loss_mean=baseline_loss_mean)

        def record_pending_index_drain():
            completion_operation_calls.append(("drain", None))
            return drain_pending_sample_indices()

        with (
            patch.object(loss_change_monitor, "reset", record_monitor_reset),
            patch.object(
                pending_training_assignment_buffer,
                "drain_pending_sample_indices",
                record_pending_index_drain,
            ),
        ):
            response_completion = complete_alarm_buffer_response(
                alarm_buffer_response=alarm_buffer_response, **completion_arguments
            )
        assert completion_operation_calls == [
            ("reset", expected_baseline_mean_loss),
            ("drain", None),
        ]
        assert response_completion.loss_monitoring_baseline_mean_loss == expected_baseline_mean_loss


def test_completion_record_is_frozen_and_rejects_inconsistent_fields(
    monkeypatch, valid_run_settings_mapping
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(823)
        response_arguments, completion_arguments, _, _, _, _ = build_response_completion_oracle(
            monkeypatch=monkeypatch,
            loss_change_detection_settings=valid_run_settings_mapping[
                "loss_change_detection_settings"
            ],
            alarm_interval_resolution_case="other_model_reused",
            earlier_sample_count=0,
        )
        reused_response_completion = complete_alarm_buffer_response(
            alarm_buffer_response=respond_to_alarm_with_buffered_samples(**response_arguments),
            **completion_arguments,
        )
    assert (
        reused_response_completion.alarm_buffer_response.response_outcome
        == "alarm_interval_held_model_reused"
    )
    assert (
        reused_response_completion.training_model_switch_sample_index
        == reused_response_completion.alarm_sample_index
    )
    assert reused_response_completion.detection_episode_operation_required is True
    with pytest.raises(FrozenInstanceError):
        reused_response_completion.alarm_sample_index = 0
    with pytest.raises(TypeError):
        AlarmResponseCompletion(*asdict(reused_response_completion).values())
    for invalid_fields, expected_exception in (
        (dict(alarm_buffer_response=object()), TypeError),
        (dict(alarm_sample_index=True), TypeError),
        (dict(alarm_sample_index=-1), ValueError),
        (dict(previous_training_model_id=True), TypeError),
        (dict(current_training_model_id=1.0), TypeError),
        (dict(estimated_change_point_sample_index=False), TypeError),
        (dict(estimated_change_point_sample_index=-1), ValueError),
        (dict(detection_episode_id=True), TypeError),
        (dict(detection_episode_id=-1), ValueError),
        (dict(loss_monitoring_baseline_mean_loss=1), TypeError),
        (dict(loss_monitoring_baseline_mean_loss=0.0), ValueError),
        (dict(loss_monitoring_baseline_mean_loss=1.0), ValueError),
        (dict(loss_monitoring_baseline_mean_loss=float("nan")), ValueError),
        (dict(drained_pending_sample_indices=[]), TypeError),
        (dict(drained_pending_sample_indices=(True,)), TypeError),
        (dict(drained_pending_sample_indices=(-1,)), ValueError),
        # 再利用の応答は帰属IDの変更を伴い、それ以外の応答は伴わない。
        (
            dict(previous_training_model_id=reused_response_completion.current_training_model_id),
            ValueError,
        ),
    ):
        with pytest.raises(expected_exception):
            replace(reused_response_completion, **invalid_fields)
