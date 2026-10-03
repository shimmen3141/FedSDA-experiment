"""単一runの実行条件・乱数・呼出順を検証する。"""

import random
import inspect
from types import SimpleNamespace
from typing import get_protocol_members
from contextlib import nullcontext
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
import torch

from test_run_settings_validation import valid_run_settings_mapping
from federated_learning_experiments.configuration.experiment_run_conditions import ExperimentRunConditions
from federated_learning_experiments.configuration.run_settings import ValidatedExperimentRunSettingsSubset
from federated_learning_experiments.core.configuration_errors import RunSettingsValidationError
from federated_learning_experiments.data.concept_schedules.random_concept_schedule_settings import RandomConceptScheduleSettings
from federated_learning_experiments.data.observed_streams import (
    ObservedSample, ClientObservedStream, ClientConceptTrace,
)
from federated_learning_experiments.execution.run_participant_contracts import (
    RunParticipantFactory, RunClientOperations, RunServerOperations, RunParticipants,
)
from federated_learning_experiments.execution.run_execution_records import (
    RunExecutionEvent, StreamProtocolRunResult,
)
from federated_learning_experiments.execution.run_execution_errors import RunExecutionError
from federated_learning_experiments.execution.stream_protocol_execution_settings import (
    StreamProtocolExecutionSettings,
    validate_stream_protocol_execution_settings,
)

from federated_learning_experiments.execution.run_random_sources import create_run_random_sources
from federated_learning_experiments.learning.models.torch_random_state_scope import (
    isolated_cpu_torch_random_state,
)


@pytest.fixture
def valid_run_result_field_values():
    """観測・評価情報・順序を含む独立した成功記録入力を返す。"""
    return {
        "observed_client_streams": (ClientObservedStream(client_id=0, observed_samples=(
            ObservedSample(feature_values=(0.125, 0.75), class_label=0),
            ObservedSample(feature_values=(0.5, 0.25), class_label=1),
        )),),
        "evaluation_concept_traces": (ClientConceptTrace(
            client_id=0, concept_ids_by_sample_index=(0, 1),
        ),),
        "generated_sample_count_per_client": 2,
        "processed_sample_count_per_client": 2,
        "unprocessed_tail_sample_count_per_client": 0,
        "synchronization_interval_count": 1,
        "execution_events": (
            RunExecutionEvent(stage_name="sample_processing", client_id=0, sample_index=0, round_index=0),
            RunExecutionEvent(stage_name="sample_processing", client_id=0, sample_index=1, round_index=0),
        ),
    }


@pytest.mark.parametrize("protocol_type,expected_protocol_member_names", [
    (RunParticipantFactory, {"validate_configuration", "prepare_run"}),
    (RunClientOperations, {
        "client_id", "process_observed_sample", "flush_pending_local_updates",
        "has_model_ready_for_server_registration", "advance_new_model_upload_wait_after_synchronization",
        "finalize_incomplete_candidate_validation",
    }),
    (RunServerOperations, {
        "record_client_states_before_synchronization", "synchronize_models", "finalize_started_communications",
    }),
])
def test_run_participant_protocols_expose_only_declared_observation_operations(
    protocol_type, expected_protocol_member_names,
):
    """宣言した操作だけを持ち、観測処理に真の概念入力を要求しない。"""
    protocol_member_names = get_protocol_members(protocol_type)
    assert protocol_member_names == expected_protocol_member_names
    operation_parameters = {
        "validate_configuration": (),
        "prepare_run": ("experiment_run_conditions", "run_random_sources", "sample_generator"),
        "process_observed_sample": ("observed_sample", "sample_index"),
        "flush_pending_local_updates": ("round_index",),
        "has_model_ready_for_server_registration": (),
        "advance_new_model_upload_wait_after_synchronization": ("round_index",),
        "finalize_incomplete_candidate_validation": (),
        "record_client_states_before_synchronization": ("round_index",),
        "synchronize_models": ("round_index", "new_model_registration_available"),
        "finalize_started_communications": ("completed_round_count",),
    }
    for operation_name in protocol_member_names - {"client_id"}:
        assert tuple(inspect.signature(getattr(protocol_type, operation_name)).parameters) == (
            "self", *operation_parameters[operation_name],
        )
        for position_parameter_name in operation_parameters[operation_name]:
            assert inspect.signature(getattr(protocol_type, operation_name)).parameters[
                position_parameter_name
            ].kind is inspect.Parameter.KEYWORD_ONLY
    with pytest.raises(TypeError):
        protocol_type()


def test_run_participants_preserve_operation_references_and_reject_mutable_client_collections():
    """接続先が所有する可変状態への参照を保持し、参加者集合だけを固定する。"""
    operation_reference = SimpleNamespace(
        client_id=0,
        process_observed_sample=lambda *, observed_sample, sample_index: None,
        flush_pending_local_updates=lambda *, round_index: None,
        has_model_ready_for_server_registration=lambda: False,
        advance_new_model_upload_wait_after_synchronization=lambda *, round_index: None,
        finalize_incomplete_candidate_validation=lambda: None,
        record_client_states_before_synchronization=lambda *, round_index: None,
        synchronize_models=lambda *, round_index, new_model_registration_available: None,
        finalize_started_communications=lambda *, completed_round_count: None,
    )
    participants = RunParticipants(
        client_operations=(operation_reference,), server_operations=operation_reference,
    )
    assert participants.client_operations[0] is operation_reference
    assert participants.server_operations is operation_reference
    for protocol_type in (RunClientOperations, RunServerOperations):
        for operation_name in get_protocol_members(protocol_type) - {"client_id"}:
            assert callable(getattr(operation_reference, operation_name))
            assert tuple(inspect.signature(getattr(operation_reference, operation_name)).parameters) == tuple(
                inspect.signature(getattr(protocol_type, operation_name)).parameters
            )[1:]
    operation_reference.client_id = 7
    assert participants.client_operations[0].client_id == 7
    for record_field_name in ("client_operations", "server_operations"):
        with pytest.raises(FrozenInstanceError):
            setattr(participants, record_field_name, None)
        with pytest.raises(FrozenInstanceError):
            delattr(participants, record_field_name)
    with pytest.raises(TypeError, match="client_operations"):
        RunParticipants(client_operations=[operation_reference], server_operations=operation_reference)
    with pytest.raises(TypeError):
        RunParticipants((operation_reference,), operation_reference)


def test_run_execution_events_preserve_stage_and_optional_positions():
    """正式な各stageと、0始まりの位置または対象のないNoneを保持する。"""
    stage_names = (
        "configuration_validation", "initial_preparation", "participant_validation",
        "concept_trace_generation", "observed_stream_generation", "sample_processing",
        "pending_update_flush", "pre_sync_recording", "registration_readiness_check",
        "server_synchronization", "upload_wait_advance",
        "incomplete_candidate_validation_finalization", "started_communication_finalization",
    )
    for stage_name in stage_names:
        execution_event = RunExecutionEvent(stage_name=stage_name)
        assert execution_event.stage_name == stage_name
        assert execution_event.client_id is execution_event.sample_index is execution_event.round_index is None
    execution_event = RunExecutionEvent(
        stage_name="sample_processing", client_id=0, sample_index=1499, round_index=29,
    )
    assert (execution_event.client_id, execution_event.sample_index, execution_event.round_index) == (0, 1499, 29)


def test_run_execution_records_are_frozen_and_keyword_only(valid_run_result_field_values):
    """成功結果とイベントの全フィールドの変更・削除・位置引数を拒否する。"""
    for record_type, record_field_values in (
        (RunExecutionEvent, {"stage_name": "sample_processing", "client_id": 0,
                            "sample_index": 0, "round_index": 0}),
        (StreamProtocolRunResult, valid_run_result_field_values),
    ):
        record_instance = record_type(**record_field_values)
        for record_field_name in record_field_values:
            with pytest.raises(FrozenInstanceError):
                setattr(record_instance, record_field_name, None)
            with pytest.raises(FrozenInstanceError):
                delattr(record_instance, record_field_name)
        with pytest.raises(TypeError):
            record_type(*record_field_values.values())


def test_stream_protocol_results_preserve_observations_truth_counts_and_event_order(
    valid_run_result_field_values,
):
    """観測・評価真値・件数・実行順を保持し、処理部や乱数参照を残さない。"""
    run_result = StreamProtocolRunResult(**valid_run_result_field_values)
    assert set(run_result.__dataclass_fields__) == set(valid_run_result_field_values)
    for record_field_name, record_field_values in valid_run_result_field_values.items():
        assert getattr(run_result, record_field_name) is record_field_values
    assert run_result.observed_client_streams[0].observed_samples[1].class_label == 1
    assert run_result.evaluation_concept_traces[0].concept_ids_by_sample_index == (0, 1)
    assert tuple(execution_event.sample_index for execution_event in run_result.execution_events) == (0, 1)
    with pytest.raises(TypeError):
        run_result.execution_events[0] = None
    with pytest.raises(FrozenInstanceError):
        run_result.observed_client_streams[0].observed_samples[0].class_label = 1


@pytest.mark.parametrize("record_field_name,invalid_field_value", [
    ("observed_client_streams", []), ("observed_client_streams", (SimpleNamespace(),)),
    ("evaluation_concept_traces", []), ("evaluation_concept_traces", (SimpleNamespace(),)),
    ("execution_events", []), ("execution_events", (SimpleNamespace(),)),
    *[(record_field_name, invalid_field_value)
      for record_field_name in ("generated_sample_count_per_client", "processed_sample_count_per_client",
                                "unprocessed_tail_sample_count_per_client", "synchronization_interval_count")
      for invalid_field_value in (-1, True, 0.0, "0", None)],
])
def test_stream_protocol_results_reject_mutable_or_invalid_records(
    valid_run_result_field_values, record_field_name, invalid_field_value,
):
    """入れ子の不変記録以外や可変集合、不正な件数を保持しない。"""
    valid_run_result_field_values[record_field_name] = invalid_field_value
    with pytest.raises((TypeError, ValueError), match=record_field_name):
        StreamProtocolRunResult(**valid_run_result_field_values)


def test_run_execution_errors_preserve_stage_positions_reason_and_cause():
    """正式stage・位置・理由と、標準causeによる元例外の参照を保持する。"""
    original_exception = ValueError("接続先の失敗")
    run_execution_error = RunExecutionError(
        stage_name="sample_processing", client_id=2, sample_index=101, round_index=2,
        failure_reason="標本処理に失敗しました。",
    )
    with pytest.raises(RunExecutionError) as exception_info:
        raise run_execution_error from original_exception
    assert exception_info.value is run_execution_error
    assert run_execution_error.__cause__ is original_exception
    assert isinstance(run_execution_error, RuntimeError)
    assert run_execution_error.stage_name == "sample_processing"
    assert (run_execution_error.client_id, run_execution_error.sample_index, run_execution_error.round_index) == (2, 101, 2)
    assert run_execution_error.failure_reason == "標本処理に失敗しました。"
    assert all(str(position_parameter_value) in str(run_execution_error)
               for position_parameter_value in ("sample_processing", 2, 101, "標本処理に失敗しました。"))
    run_execution_error = RunExecutionError(stage_name="initial_preparation", failure_reason="準備に失敗しました。")
    assert run_execution_error.client_id is run_execution_error.sample_index is run_execution_error.round_index is None


@pytest.mark.parametrize("record_type", [RunExecutionEvent, RunExecutionError])
@pytest.mark.parametrize("record_field_name,invalid_field_value", [
    ("stage_name", "unknown"), ("stage_name", True),
    *[(record_field_name, invalid_field_value)
      for record_field_name in ("client_id", "sample_index", "round_index")
      for invalid_field_value in (-1, True, 0.0, "0")],
])
def test_run_execution_records_reject_unknown_stages_and_invalid_positions(
    record_type, record_field_name, invalid_field_value,
):
    """不明stageと非負整数以外の位置を、イベントと例外で共通に拒否する。"""
    record_field_values = {"stage_name": "sample_processing"}
    if record_type is RunExecutionError:
        record_field_values["failure_reason"] = "標本処理に失敗しました。"
    record_field_values[record_field_name] = invalid_field_value
    with pytest.raises((TypeError, ValueError), match=record_field_name):
        record_type(**record_field_values)


def test_run_random_sources_are_independent_and_reproduce_legacy_sequences():
    global_python_random_state = random.getstate()
    global_numpy_random_state = np.random.get_state()
    run_random_sources = create_run_random_sources(random_seed=17)
    repeated_run_random_sources = create_run_random_sources(random_seed=17)
    expected_python_random_generator = random.Random(17)
    expected_numpy_random_generator = np.random.RandomState(17)
    assert run_random_sources.python_random_generator is not repeated_run_random_sources.python_random_generator
    assert run_random_sources.numpy_random_generator is not repeated_run_random_sources.numpy_random_generator
    assert run_random_sources.python_random_generator.random() == expected_python_random_generator.random()
    np.testing.assert_array_equal(
        run_random_sources.numpy_random_generator.uniform(size=10),
        expected_numpy_random_generator.uniform(size=10),
    )
    assert repeated_run_random_sources.python_random_generator.random() == random.Random(17).random()
    np.testing.assert_array_equal(
        repeated_run_random_sources.numpy_random_generator.uniform(size=10),
        np.random.RandomState(17).uniform(size=10),
    )
    assert random.getstate() == global_python_random_state
    np.testing.assert_equal(np.random.get_state(), global_numpy_random_state)


@pytest.mark.parametrize("fail_inside_scope", [False, True])
def test_cpu_torch_random_scope_restores_state_on_success_and_failure(fail_inside_scope):
    original_cpu_torch_random_state = torch.get_rng_state().clone()
    expected_cpu_random_generator = torch.Generator(device="cpu").manual_seed(17)
    with pytest.raises(ValueError, match="テスト内の失敗") if fail_inside_scope else nullcontext():
        with isolated_cpu_torch_random_state(random_seed=17):
            observed_cpu_random_values = torch.rand(10)
            assert torch.equal(
                observed_cpu_random_values,
                torch.rand(10, generator=expected_cpu_random_generator),
            )
            if fail_inside_scope:
                raise ValueError("テスト内の失敗")
    assert torch.equal(torch.get_rng_state(), original_cpu_torch_random_state)


def test_cpu_torch_random_scope_does_not_seed_other_devices_or_change_runtime_defaults(monkeypatch):
    non_cpu_seed_calls = []
    original_torch_thread_count = torch.get_num_threads()
    original_torch_default_dtype = torch.get_default_dtype()
    monkeypatch.setattr(torch, "manual_seed", non_cpu_seed_calls.append)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", non_cpu_seed_calls.append)
    monkeypatch.setattr(torch.xpu, "manual_seed_all", non_cpu_seed_calls.append)
    with isolated_cpu_torch_random_state(random_seed=0):
        torch.rand(3)
    assert non_cpu_seed_calls == []
    assert torch.get_num_threads() == original_torch_thread_count
    assert torch.get_default_dtype() == original_torch_default_dtype


@pytest.fixture
def valid_stream_protocol_execution_settings():
    return StreamProtocolExecutionSettings(
        experiment_run_conditions=ExperimentRunConditions(
            dataset_name="sine2", random_seed=0, client_count=3,
            per_client_sample_count=1500, server_aggregation_interval_per_client_samples=50,
        ),
        concept_schedule_settings=RandomConceptScheduleSettings(
            concept_schedule_strategy="random_changes_after_minimum_index_gap",
            minimum_sample_index_gap_before_change_trial=100,
            per_eligible_sample_concept_change_probability=0.015,
        ),
        execution_strategy="sample_index_then_client_order_with_interval_synchronization",
    )


@pytest.mark.parametrize("random_seed", [0, 2**32 - 1])
@pytest.mark.parametrize("per_client_sample_count,server_aggregation_interval_per_client_samples", [(16, 7), (3, 7)])
def test_stream_protocol_execution_settings_accept_seed_and_interval_boundaries(
    valid_stream_protocol_execution_settings, random_seed, per_client_sample_count,
    server_aggregation_interval_per_client_samples,
):
    execution_settings = replace(
        valid_stream_protocol_execution_settings,
        experiment_run_conditions=replace(
            valid_stream_protocol_execution_settings.experiment_run_conditions,
            random_seed=random_seed, per_client_sample_count=per_client_sample_count,
            server_aggregation_interval_per_client_samples=server_aggregation_interval_per_client_samples,
        ),
    )
    assert validate_stream_protocol_execution_settings(execution_settings=execution_settings) is None


@pytest.mark.parametrize("configuration_parameter_name,specified_parameter_value", [
    ("experiment_run_conditions", None), ("concept_schedule_settings", {}),
    ("execution_strategy", "unknown"), ("execution_strategy", None),
    ("execution_strategy", np.array(["sample_index_then_client_order_with_interval_synchronization"])),
    ("dataset_name", "sea2"), ("dataset_name", "mnist2"), ("random_seed", 2**32),
])
def test_stream_protocol_execution_settings_reject_invalid_components_and_choices(
    valid_stream_protocol_execution_settings, configuration_parameter_name, specified_parameter_value,
):
    with pytest.raises(RunSettingsValidationError) as exception_info:
        if configuration_parameter_name in ("dataset_name", "random_seed"):
            replace(
                valid_stream_protocol_execution_settings,
                experiment_run_conditions=replace(
                    valid_stream_protocol_execution_settings.experiment_run_conditions,
                    **{configuration_parameter_name: specified_parameter_value},
                ),
            )
        else:
            replace(valid_stream_protocol_execution_settings, **{configuration_parameter_name: specified_parameter_value})
    assert exception_info.value.configuration_parameter_name == configuration_parameter_name
    assert exception_info.value.specified_parameter_value == specified_parameter_value
    assert exception_info.value.validation_failure_reason


def test_stream_protocol_execution_validation_rejects_partial_settings(valid_run_settings_mapping):
    partial_run_settings = ValidatedExperimentRunSettingsSubset(**valid_run_settings_mapping)
    with pytest.raises(RunSettingsValidationError) as exception_info:
        validate_stream_protocol_execution_settings(execution_settings=partial_run_settings)
    assert exception_info.value.configuration_parameter_name == "execution_settings"
    assert exception_info.value.specified_parameter_value is partial_run_settings


@pytest.mark.parametrize("missing_parameter_name", [
    "experiment_run_conditions", "concept_schedule_settings", "execution_strategy",
])
def test_stream_protocol_execution_settings_are_required_frozen_and_keyword_only(
    valid_stream_protocol_execution_settings, missing_parameter_name,
):
    with pytest.raises(TypeError):
        StreamProtocolExecutionSettings(**{
            configuration_parameter_name: specified_parameter_value
            for configuration_parameter_name, specified_parameter_value
            in vars(valid_stream_protocol_execution_settings).items()
            if configuration_parameter_name != missing_parameter_name
        })
    with pytest.raises(TypeError):
        StreamProtocolExecutionSettings(*vars(valid_stream_protocol_execution_settings).values())
    with pytest.raises(FrozenInstanceError):
        setattr(valid_stream_protocol_execution_settings, missing_parameter_name, None)
    with pytest.raises(FrozenInstanceError):
        delattr(valid_stream_protocol_execution_settings, missing_parameter_name)
