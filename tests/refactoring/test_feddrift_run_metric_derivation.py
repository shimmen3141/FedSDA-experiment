"""共通指標を実旧全体runへ照合し、読取り性と不整合拒否を検査する。"""

from copy import deepcopy
from dataclasses import replace
from math import isnan

import pytest

from federated_learning_experiments.evaluation.run_metric_calculations import RunMetricSettings
from federated_learning_experiments.runtime.feddrift_run_metric_derivation import (
    derive_feddrift_run_metrics,
)
from federated_learning_experiments.runtime.feddrift_run_participant_factory import (
    FeddriftRunParticipantFactory,
)
from federated_learning_experiments.runtime.feddrift_stream_run_execution import (
    execute_feddrift_stream_protocol_run,
)
from tests.refactoring.test_feddrift_run_client import assert_rng_equal, rng_snapshot
from tests.refactoring.test_feddrift_stream_finalization import (
    assert_terminal_values_equal,
    terminal_snapshot,
)
from tests.refactoring.test_feddrift_stream_run_execution import run_inputs, run_legacy

METRIC_SETTINGS = RunMetricSettings(
    maximum_detection_delay_sample_count=3, post_change_recovery_window_sample_count=3
)


def basic_metric_values(metrics):
    """旧成果物の各fieldへ新APIを明示的に対応付ける。"""
    detection = metrics.training_model_switch_detection_metrics
    values = dict(
        accuracy=metrics.prediction_accuracy,
        stable_accuracy=metrics.stable_period_prediction_accuracy,
        precision=detection.detection_precision,
        recall=detection.detection_recall,
        f1=detection.detection_f1,
        total_detect=detection.detection_count,
        final_model_count=metrics.final_global_model_count,
        final_parameter_values=metrics.final_parameter_value_count,
        final_parameter_bytes=metrics.final_parameter_byte_count,
    )
    for old, new in (
        ("comm_models_up", "uploaded_model_count"),
        ("comm_models_down", "downloaded_model_count"),
        ("comm_messages_up", "uploaded_message_count"),
        ("comm_messages_down", "downloaded_message_count"),
        ("comm_parameter_values_up", "uploaded_parameter_value_count"),
        ("comm_parameter_values_down", "downloaded_parameter_value_count"),
        ("comm_bytes_up", "uploaded_byte_count"),
        ("comm_bytes_down", "downloaded_byte_count"),
    ):
        values[old] = getattr(metrics.communication_volume, new)
    return values


@pytest.mark.parametrize("dataset", ("sine2", "sea4", "mnist2"))
@pytest.mark.parametrize(
    "optimizer,threshold,count", (("sgd", 1.0, 11), ("adam", 0.0, 11), ("sgd", 0.03, 2))
)
def test_basic_metrics_match_actual_legacy_and_are_read_only(
    monkeypatch, dataset, optimizer, threshold, count
):
    settings, run_settings = run_inputs(monkeypatch, dataset, optimizer, count, threshold)
    old = run_legacy(monkeypatch, run_settings, threshold)
    factory = FeddriftRunParticipantFactory(run_participant_settings=settings)
    result = execute_feddrift_stream_protocol_run(
        execution_settings=run_settings, participant_factory=factory
    )
    participants = factory.prepared_run_participants
    before = deepcopy(terminal_snapshot(participants.run_server))
    rng = rng_snapshot()
    metrics = derive_feddrift_run_metrics(
        run_result=result, participants=participants, run_metric_settings=METRIC_SETTINGS
    )
    for key, value in basic_metric_values(metrics).items():
        if isinstance(value, float) and isnan(value):
            assert isnan(old["metrics"][key])
        else:
            assert value == old["metrics"][key], (key, value, old["metrics"][key])
    assert_terminal_values_equal(terminal_snapshot(participants.run_server), before)
    assert_rng_equal(rng)


@pytest.mark.parametrize(
    "invalid",
    (
        "result_type",
        "participants_type",
        "settings_type",
        "settings_value",
        "trace_id",
        "stream_id",
        "stream_length",
        "generated_count",
        "processed_count",
        "tail_count",
        "round_count",
        "prediction_count",
        "prediction_index",
        "prediction_label",
        "prediction_concept",
        "prediction_type",
        "concept_bool",
        "stream_label_bool",
        "client_id",
        "event_stage",
        "transition_type",
        "transition_count",
        "transition_position",
        "prediction_transition",
    ),
)
def test_metrics_reject_inconsistent_inputs_without_updates(monkeypatch, invalid):
    settings, run_settings = run_inputs(monkeypatch, count=5)
    factory = FeddriftRunParticipantFactory(run_participant_settings=settings)
    result = execute_feddrift_stream_protocol_run(
        execution_settings=run_settings, participant_factory=factory
    )
    participants = factory.prepared_run_participants
    metric_settings = replace(METRIC_SETTINGS)
    if invalid == "result_type":
        result = object()
    elif invalid == "participants_type":
        participants = object()
    elif invalid == "settings_type":
        metric_settings = object()
    elif invalid == "settings_value":
        object.__setattr__(metric_settings, "maximum_detection_delay_sample_count", True)
    elif invalid == "trace_id":
        object.__setattr__(result.evaluation_concept_traces[0], "client_id", 1)
    elif invalid == "stream_id":
        object.__setattr__(result.observed_client_streams[0], "client_id", 1)
    elif invalid == "stream_length":
        object.__setattr__(result.observed_client_streams[0], "observed_samples", ())
    elif invalid == "generated_count":
        object.__setattr__(result, "generated_sample_count_per_client", 4)
    elif invalid == "processed_count":
        object.__setattr__(result, "processed_sample_count_per_client", 2)
    elif invalid == "tail_count":
        object.__setattr__(result, "unprocessed_tail_sample_count_per_client", 1)
    elif invalid == "round_count":
        object.__setattr__(result, "synchronization_interval_count", 2)
    elif invalid == "prediction_count":
        participants.run_clients[0].stream_processor._predictions.pop()
    elif invalid == "prediction_index":
        object.__setattr__(
            participants.run_clients[0].stream_processor._predictions[0], "sample_index", 1
        )
    elif invalid == "prediction_label":
        object.__setattr__(
            participants.run_clients[0].stream_processor._predictions[0], "observed_class_id", 4
        )
    elif invalid == "prediction_concept":
        object.__setattr__(
            participants.run_clients[0].stream_processor._predictions[0], "observed_concept_id", 7
        )
    elif invalid == "prediction_type":
        participants.run_clients[0].stream_processor._predictions[0] = None
    elif invalid == "concept_bool":
        object.__setattr__(
            result.evaluation_concept_traces[0], "concept_ids_by_sample_index", (True,) * 5
        )
    elif invalid == "stream_label_bool":
        object.__setattr__(
            result.observed_client_streams[0].observed_samples[0], "class_label", True
        )
    elif invalid == "client_id":
        participants.run_clients[0]._client_id = True
    elif invalid == "event_stage":
        object.__setattr__(result.execution_events[0], "stage_name", "wrong")
    elif invalid == "transition_type":
        participants.run_clients[0].stream_processor._transitions[0] = None
    elif invalid == "transition_count":
        participants.run_clients[0].stream_processor._transitions.clear()
    elif invalid == "transition_position":
        object.__setattr__(
            participants.run_clients[0].stream_processor._transitions[0], "first_sample_index", 1
        )
    elif invalid == "prediction_transition":
        object.__setattr__(
            participants.run_clients[0].stream_processor._predictions[2],
            "detection_batch_transition",
            None,
        )
    # 壊した入力自体は保存し、導出による追加変更だけを検出する。
    server_before = deepcopy(terminal_snapshot(factory.prepared_run_participants.run_server))
    rng = rng_snapshot()
    with pytest.raises((TypeError, ValueError)):
        derive_feddrift_run_metrics(
            run_result=result, participants=participants, run_metric_settings=metric_settings
        )
    assert_terminal_values_equal(
        terminal_snapshot(factory.prepared_run_participants.run_server), server_before
    )
    assert_rng_equal(rng)


def test_tail_concept_changes_count_as_missed(monkeypatch):
    settings, run_settings = run_inputs(monkeypatch, count=2, probability=1.0)
    run_settings = replace(
        run_settings,
        concept_schedule_settings=replace(
            run_settings.concept_schedule_settings, minimum_sample_index_gap_before_change_trial=0
        ),
    )
    factory = FeddriftRunParticipantFactory(run_participant_settings=settings)
    result = execute_feddrift_stream_protocol_run(
        execution_settings=run_settings, participant_factory=factory
    )
    metrics = derive_feddrift_run_metrics(
        run_result=result,
        participants=factory.prepared_run_participants,
        run_metric_settings=METRIC_SETTINGS,
    )
    assert metrics.prediction_accuracy == 0.0
    assert isnan(metrics.stable_period_prediction_accuracy)
    assert metrics.training_model_switch_detection_metrics.concept_change_count == 2
    assert metrics.training_model_switch_detection_metrics.detection_recall == 0.0
