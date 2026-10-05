"""単一runの実行条件・乱数・呼出順を検証する。"""

import random
import inspect
from types import SimpleNamespace
from typing import get_protocol_members
from contextlib import nullcontext
from dataclasses import FrozenInstanceError, replace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from test_run_settings_validation import valid_run_settings_mapping
from test_sine_stream_generation import assert_numpy_random_states_equal
from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.configuration.run_settings import (
    ValidatedExperimentRunSettingsSubset,
)
from federated_learning_experiments.core.configuration_errors import RunSettingsValidationError
from federated_learning_experiments.data.concept_schedules.random_concept_schedule_settings import (
    RandomConceptScheduleSettings,
)
from federated_learning_experiments.data.observed_streams import (
    ObservedSample,
    ClientObservedStream,
    ClientConceptTrace,
)
from federated_learning_experiments.execution.run_participant_contracts import (
    RunParticipantFactory,
    RunClientOperations,
    RunServerOperations,
    RunParticipants,
)
from federated_learning_experiments.execution.run_execution_records import (
    RunExecutionEvent,
    StreamProtocolRunResult,
)
from federated_learning_experiments.execution.run_execution_errors import RunExecutionError
from federated_learning_experiments.execution.stream_protocol_execution_loop import (
    invoke_run_operation_and_record_success,
    run_stream_protocol_intervals,
)
from federated_learning_experiments.execution.stream_protocol_execution_settings import (
    StreamProtocolExecutionSettings,
    validate_stream_protocol_execution_settings,
)

from federated_learning_experiments.execution.run_random_sources import create_run_random_sources
from federated_learning_experiments.learning.models.torch_random_state_scope import (
    isolated_cpu_torch_random_state,
)
from federated_learning_experiments.runtime import single_run_execution as runtime_module


class RunOperationObserver:
    """操作試行と故障を記録する共有状態。研究手法の計算を行わない。"""

    def __init__(self):
        self.operation_calls: list[RunExecutionEvent] = []
        self.processed_observed_samples: list[tuple[int, int, ObservedSample]] = []
        self.synchronization_readiness_values: list[bool] = []
        self.registration_readiness_by_client: dict[int, object] = {}
        self.failure_at_event: RunExecutionEvent | None = None
        self.injected_exception: Exception | None = None
        self.current_round_index: int | None = None
        self.finalized_communication_round_counts: list[int] = []

    def record_operation_call(
        self, *, stage_name, client_id=None, sample_index=None, round_index=None
    ):
        execution_event = RunExecutionEvent(
            stage_name=stage_name,
            client_id=client_id,
            sample_index=sample_index,
            round_index=round_index,
        )
        self.operation_calls.append(execution_event)
        if execution_event == self.failure_at_event and self.injected_exception is not None:
            raise self.injected_exception


class ObservingRunClient:
    """client操作を共有observerへ記録するテスト専用接続先。"""

    def __init__(self, *, client_id, observer):
        self.client_id = client_id
        self.observer = observer

    def process_observed_sample(self, *, observed_sample, sample_index):
        self.observer.record_operation_call(
            stage_name="sample_processing",
            client_id=self.client_id,
            sample_index=sample_index,
            round_index=self.observer.current_round_index,
        )
        self.observer.processed_observed_samples.append(
            (self.client_id, sample_index, observed_sample)
        )

    def flush_pending_local_updates(self, *, round_index):
        self.observer.record_operation_call(
            stage_name="pending_update_flush",
            client_id=self.client_id,
            round_index=round_index,
        )

    def has_model_ready_for_server_registration(self):
        self.observer.record_operation_call(
            stage_name="registration_readiness_check",
            client_id=self.client_id,
            round_index=self.observer.current_round_index,
        )
        return self.observer.registration_readiness_by_client.get(self.client_id, False)

    def advance_new_model_upload_wait_after_synchronization(self, *, round_index):
        self.observer.record_operation_call(
            stage_name="upload_wait_advance",
            client_id=self.client_id,
            round_index=round_index,
        )
        # 次の標本処理を観測するときの区間位置だけを進める。
        self.observer.current_round_index = round_index + 1

    def finalize_incomplete_candidate_validation(self):
        self.observer.record_operation_call(
            stage_name="incomplete_candidate_validation_finalization",
            client_id=self.client_id,
        )


class ObservingRunServer:
    """同期時のboolと通信終端の完了区間数を観測する接続先。"""

    def __init__(self, *, observer):
        self.observer = observer

    def record_client_states_before_synchronization(self, *, round_index):
        self.observer.current_round_index = round_index
        self.observer.record_operation_call(
            stage_name="pre_sync_recording", round_index=round_index
        )

    def synchronize_models(self, *, round_index, new_model_registration_available):
        self.observer.record_operation_call(
            stage_name="server_synchronization", round_index=round_index
        )
        self.observer.synchronization_readiness_values.append(new_model_registration_available)

    def finalize_started_communications(self, *, completed_round_count):
        self.observer.record_operation_call(stage_name="started_communication_finalization")
        self.observer.finalized_communication_round_counts.append(completed_round_count)


def build_observing_run_participants(*, client_count, observer):
    """runごとの新しい観測用参加者を作る。"""
    observer.current_round_index = 0
    return RunParticipants(
        client_operations=tuple(
            ObservingRunClient(client_id=client_id, observer=observer)
            for client_id in range(client_count)
        ),
        server_operations=ObservingRunServer(observer=observer),
    )


def build_interval_test_observed_streams(*, client_count, per_client_sample_count):
    """client・位置を特徴から識別できる不変の観測列を作る。"""
    return tuple(
        ClientObservedStream(
            client_id=client_id,
            observed_samples=tuple(
                ObservedSample(
                    feature_values=(float(client_id), float(sample_index)),
                    class_label=sample_index % 2,
                )
                for sample_index in range(per_client_sample_count)
            ),
        )
        for client_id in range(client_count)
    )


class ObservingRunParticipantFactory:
    """初期準備の試行と新規参加者だけを観測するテスト専用factory。"""

    def __init__(self, *, observer):
        self.observer = observer
        self.prepared_participant_history: list[RunParticipants] = []
        self.borrowed_run_random_sources = None
        self.borrowed_sample_generator = None

    def validate_configuration(self):
        """この観測用factoryには追加の固定条件がない。"""

    def prepare_run(self, *, experiment_run_conditions, run_random_sources, sample_generator):
        self.observer.record_operation_call(stage_name="initial_preparation")
        self.borrowed_run_random_sources = run_random_sources
        self.borrowed_sample_generator = sample_generator
        participants = build_observing_run_participants(
            client_count=experiment_run_conditions.client_count,
            observer=self.observer,
        )
        self.prepared_participant_history.append(participants)
        return participants


class InitialPreparationObservingParticipantFactory(ObservingRunParticipantFactory):
    """100標本と10shuffleの準備消費だけを再現し、学習を実装しない。"""

    def prepare_run(self, *, experiment_run_conditions, run_random_sources, sample_generator):
        participants = super().prepare_run(
            experiment_run_conditions=experiment_run_conditions,
            run_random_sources=run_random_sources,
            sample_generator=sample_generator,
        )
        self.preparation_observed_samples = [
            sample_generator.generate_sample(concept_id=0)
            for preparation_sample_index in range(100)
        ]
        for preparation_shuffle_index in range(10):
            run_random_sources.python_random_generator.shuffle(self.preparation_observed_samples)
        self.prepared_python_random_state = run_random_sources.python_random_generator.getstate()
        self.prepared_numpy_random_state = run_random_sources.numpy_random_generator.get_state()
        return participants


def build_reference_post_preparation_streams(*, execution_settings, monkeypatch):
    """独立乱数を旧helperへ注入して準備後データと二時点の状態を返す。"""
    from federated_drift_experiment.data import schedules as reference_schedules_module
    from federated_drift_experiment.data import synthetic as reference_synthetic_module

    experiment_run_conditions = execution_settings.experiment_run_conditions
    concept_schedule_settings = execution_settings.concept_schedule_settings
    reference_python_random_generator = random.Random(experiment_run_conditions.random_seed)
    reference_numpy_random_generator = np.random.RandomState(experiment_run_conditions.random_seed)
    monkeypatch.setattr(reference_schedules_module, "random", reference_python_random_generator)
    monkeypatch.setattr(
        reference_synthetic_module,
        "np",
        SimpleNamespace(
            random=reference_numpy_random_generator,
            sin=np.sin,
        ),
    )
    reference_feature_values, reference_class_labels = reference_synthetic_module.generate_sine2(
        0, 100
    )
    reference_preparation_samples = list(
        zip(reference_feature_values, reference_class_labels, strict=True)
    )
    for preparation_shuffle_index in range(10):
        reference_python_random_generator.shuffle(reference_preparation_samples)
    reference_prepared_python_random_state = reference_python_random_generator.getstate()
    reference_prepared_numpy_random_state = reference_numpy_random_generator.get_state()
    reference_concept_schedules = reference_schedules_module.make_random_schedules(
        experiment_run_conditions.client_count,
        experiment_run_conditions.per_client_sample_count,
        2,
        concept_schedule_settings.minimum_sample_index_gap_before_change_trial,
        concept_schedule_settings.per_eligible_sample_concept_change_probability,
    )
    evaluation_concept_traces = tuple(
        ClientConceptTrace(
            client_id=client_id,
            concept_ids_by_sample_index=tuple(concept_ids_by_sample_index),
        )
        for client_id, concept_ids_by_sample_index in enumerate(reference_concept_schedules)
    )
    observed_client_streams = []
    for client_concept_trace in evaluation_concept_traces:
        reference_observed_samples = []
        for concept_id in client_concept_trace.concept_ids_by_sample_index:
            reference_feature_values, reference_class_labels = (
                reference_synthetic_module.generate_sine2(concept_id, 1)
            )
            reference_observed_samples.append(
                ObservedSample(
                    feature_values=tuple(torch.FloatTensor(reference_feature_values[0]).tolist()),
                    class_label=int(reference_class_labels[0]),
                )
            )
        observed_client_streams.append(
            ClientObservedStream(
                client_id=client_concept_trace.client_id,
                observed_samples=tuple(reference_observed_samples),
            )
        )
    reference_generated_python_random_state = reference_python_random_generator.getstate()
    reference_generated_numpy_random_state = reference_numpy_random_generator.get_state()
    return (
        evaluation_concept_traces,
        tuple(observed_client_streams),
        reference_prepared_python_random_state,
        reference_prepared_numpy_random_state,
        reference_generated_python_random_state,
        reference_generated_numpy_random_state,
    )


@pytest.mark.parametrize("random_seed", [0, 17])
@pytest.mark.parametrize("per_eligible_sample_concept_change_probability", [0.0, 0.015, 1.0])
def test_single_run_post_preparation_streams_match_reference(
    monkeypatch,
    valid_stream_protocol_execution_settings,
    random_seed,
    per_eligible_sample_concept_change_probability,
):
    """100標本・10shuffle後の全データと準備直後・供給後乱数を旧基準へ照合する。"""
    execution_settings = replace(
        valid_stream_protocol_execution_settings,
        experiment_run_conditions=replace(
            valid_stream_protocol_execution_settings.experiment_run_conditions,
            random_seed=random_seed,
        ),
        concept_schedule_settings=replace(
            valid_stream_protocol_execution_settings.concept_schedule_settings,
            per_eligible_sample_concept_change_probability=per_eligible_sample_concept_change_probability,
        ),
    )
    global_python_random_state = random.getstate()
    global_numpy_random_state = np.random.get_state()
    original_cpu_torch_random_state = torch.get_rng_state().clone()
    participant_factory = InitialPreparationObservingParticipantFactory(
        observer=RunOperationObserver()
    )
    run_result = runtime_module.execute_stream_protocol_run(
        execution_settings=execution_settings,
        participant_factory=participant_factory,
    )
    (
        evaluation_concept_traces,
        observed_client_streams,
        reference_prepared_python_random_state,
        reference_prepared_numpy_random_state,
        reference_generated_python_random_state,
        reference_generated_numpy_random_state,
    ) = build_reference_post_preparation_streams(
        execution_settings=execution_settings, monkeypatch=monkeypatch
    )
    generated_python_random_state = (
        participant_factory.borrowed_run_random_sources.python_random_generator.getstate()
    )
    generated_numpy_random_state = (
        participant_factory.borrowed_run_random_sources.numpy_random_generator.get_state()
    )
    assert len(participant_factory.preparation_observed_samples) == 100
    assert run_result.evaluation_concept_traces == evaluation_concept_traces
    assert run_result.observed_client_streams == observed_client_streams
    assert (
        participant_factory.prepared_python_random_state == reference_prepared_python_random_state
    )
    assert_numpy_random_states_equal(
        participant_factory.prepared_numpy_random_state, reference_prepared_numpy_random_state
    )
    assert generated_python_random_state == reference_generated_python_random_state
    assert_numpy_random_states_equal(
        generated_numpy_random_state, reference_generated_numpy_random_state
    )
    assert random.getstate() == global_python_random_state
    assert_numpy_random_states_equal(np.random.get_state(), global_numpy_random_state)
    assert torch.equal(torch.get_rng_state(), original_cpu_torch_random_state)


def test_single_run_repeated_conditions_are_independent_across_intervening_runs(
    valid_stream_protocol_execution_settings,
):
    """同factoryでA→B→Aを実行し、同値結果と新しい参加者・乱数を確認する。"""
    global_python_random_state = random.getstate()
    global_numpy_random_state = np.random.get_state()
    original_cpu_torch_random_state = torch.get_rng_state().clone()
    observer = RunOperationObserver()
    participant_factory = InitialPreparationObservingParticipantFactory(observer=observer)
    first_run_result = runtime_module.execute_stream_protocol_run(
        execution_settings=valid_stream_protocol_execution_settings,
        participant_factory=participant_factory,
    )
    first_participants = participant_factory.prepared_participant_history[-1]
    first_run_random_sources = participant_factory.borrowed_run_random_sources
    intervening_execution_settings = replace(
        valid_stream_protocol_execution_settings,
        experiment_run_conditions=replace(
            valid_stream_protocol_execution_settings.experiment_run_conditions,
            random_seed=17,
            client_count=2,
            per_client_sample_count=17,
            server_aggregation_interval_per_client_samples=7,
        ),
        concept_schedule_settings=replace(
            valid_stream_protocol_execution_settings.concept_schedule_settings,
            minimum_sample_index_gap_before_change_trial=0,
            per_eligible_sample_concept_change_probability=1.0,
        ),
    )
    intervening_run_result = runtime_module.execute_stream_protocol_run(
        execution_settings=intervening_execution_settings,
        participant_factory=participant_factory,
    )
    intervening_participants = participant_factory.prepared_participant_history[-1]
    intervening_run_random_sources = participant_factory.borrowed_run_random_sources
    repeated_run_result = runtime_module.execute_stream_protocol_run(
        execution_settings=valid_stream_protocol_execution_settings,
        participant_factory=participant_factory,
    )
    repeated_participants = participant_factory.prepared_participant_history[-1]
    repeated_run_random_sources = participant_factory.borrowed_run_random_sources
    assert first_run_result == repeated_run_result
    assert first_run_result != intervening_run_result
    assert len(participant_factory.prepared_participant_history) == 3
    assert first_participants is not intervening_participants
    assert first_participants is not repeated_participants
    assert intervening_participants is not repeated_participants
    assert first_participants.server_operations is not repeated_participants.server_operations
    assert first_participants.server_operations is not intervening_participants.server_operations
    assert intervening_participants.server_operations is not repeated_participants.server_operations
    for client_id in range(2):
        assert (
            first_participants.client_operations[client_id]
            is not intervening_participants.client_operations[client_id]
        )
        assert (
            first_participants.client_operations[client_id]
            is not repeated_participants.client_operations[client_id]
        )
        assert (
            intervening_participants.client_operations[client_id]
            is not repeated_participants.client_operations[client_id]
        )
    assert first_participants.client_operations[2] is not repeated_participants.client_operations[2]
    for record_field_name in ("python_random_generator", "numpy_random_generator"):
        assert getattr(first_run_random_sources, record_field_name) is not getattr(
            intervening_run_random_sources, record_field_name
        )
        assert getattr(first_run_random_sources, record_field_name) is not getattr(
            repeated_run_random_sources, record_field_name
        )
        assert getattr(intervening_run_random_sources, record_field_name) is not getattr(
            repeated_run_random_sources, record_field_name
        )
    assert random.getstate() == global_python_random_state
    assert_numpy_random_states_equal(np.random.get_state(), global_numpy_random_state)
    assert torch.equal(torch.get_rng_state(), original_cpu_torch_random_state)


@pytest.mark.parametrize(
    "injected_failure_stage_name,expected_failure_client_id,expected_failure_sample_index,expected_failure_round_index",
    [
        ("initial_preparation", None, None, None),
        ("participant_validation", None, None, None),
        ("concept_trace_generation", None, None, None),
        ("observed_stream_generation", None, None, None),
        ("sample_processing", 1, 2, 1),
        ("pending_update_flush", 1, None, 1),
        ("pre_sync_recording", None, None, 1),
        ("registration_readiness_check", 1, None, 1),
        ("server_synchronization", None, None, 1),
        ("upload_wait_advance", 1, None, 1),
        ("incomplete_candidate_validation_finalization", 1, None, None),
        ("started_communication_finalization", None, None, None),
    ],
)
def test_single_run_failure_preserves_cause_position_and_caller_random_states(
    monkeypatch,
    valid_stream_protocol_execution_settings,
    injected_failure_stage_name,
    expected_failure_client_id,
    expected_failure_sample_index,
    expected_failure_round_index,
):
    """各故障stageで位置・元cause・全global乱数復元と停止prefixを確認する。"""
    execution_settings = replace(
        valid_stream_protocol_execution_settings,
        experiment_run_conditions=replace(
            valid_stream_protocol_execution_settings.experiment_run_conditions,
            per_client_sample_count=4,
            server_aggregation_interval_per_client_samples=2,
        ),
    )
    observer = RunOperationObserver()
    runtime_module.execute_stream_protocol_run(
        execution_settings=execution_settings,
        participant_factory=InitialPreparationObservingParticipantFactory(observer=observer),
    )
    expected_operation_calls = list(observer.operation_calls)
    observer = RunOperationObserver()
    original_exception = ValueError("統合テストの故障")
    expected_failure_event = RunExecutionEvent(
        stage_name=injected_failure_stage_name,
        client_id=expected_failure_client_id,
        sample_index=expected_failure_sample_index,
        round_index=expected_failure_round_index,
    )
    observer.failure_at_event = expected_failure_event
    observer.injected_exception = original_exception
    runtime_operation_spies = {"order": Mock()}
    for operation_name in (
        "validate_prepared_run_participants",
        "generate_random_client_concept_traces",
        "build_sine_client_observed_streams",
        "run_stream_protocol_intervals",
    ):
        run_operation = getattr(runtime_module, operation_name)
        runtime_operation_spies[operation_name] = (
            Mock(side_effect=original_exception)
            if operation_name
            == {
                "participant_validation": "validate_prepared_run_participants",
                "concept_trace_generation": "generate_random_client_concept_traces",
                "observed_stream_generation": "build_sine_client_observed_streams",
            }.get(injected_failure_stage_name)
            else Mock(wraps=run_operation)
        )
        runtime_operation_spies["order"].attach_mock(
            runtime_operation_spies[operation_name], operation_name
        )
        monkeypatch.setattr(runtime_module, operation_name, runtime_operation_spies[operation_name])
    global_python_random_state = random.getstate()
    global_numpy_random_state = np.random.get_state()
    original_cpu_torch_random_state = torch.get_rng_state().clone()
    participant_factory = InitialPreparationObservingParticipantFactory(observer=observer)
    with pytest.raises(RunExecutionError) as exception_info:
        runtime_module.execute_stream_protocol_run(
            execution_settings=execution_settings, participant_factory=participant_factory
        )
    assert exception_info.value.__cause__ is original_exception
    assert exception_info.value.stage_name == injected_failure_stage_name
    assert exception_info.value.client_id == expected_failure_client_id
    assert exception_info.value.sample_index == expected_failure_sample_index
    assert exception_info.value.round_index == expected_failure_round_index
    assert random.getstate() == global_python_random_state
    assert_numpy_random_states_equal(np.random.get_state(), global_numpy_random_state)
    assert torch.equal(torch.get_rng_state(), original_cpu_torch_random_state)
    runtime_call_order = [
        operation_call[0] for operation_call in runtime_operation_spies["order"].mock_calls
    ]
    if injected_failure_stage_name == "initial_preparation":
        assert runtime_call_order == []
    elif injected_failure_stage_name in (
        "participant_validation",
        "concept_trace_generation",
        "observed_stream_generation",
    ):
        assert (
            runtime_call_order
            == [
                "validate_prepared_run_participants",
                "generate_random_client_concept_traces",
                "build_sine_client_observed_streams",
            ][
                : [
                    "participant_validation",
                    "concept_trace_generation",
                    "observed_stream_generation",
                ].index(injected_failure_stage_name)
                + 1
            ]
        )
    else:
        assert runtime_call_order == [
            "validate_prepared_run_participants",
            "generate_random_client_concept_traces",
            "build_sine_client_observed_streams",
            "run_stream_protocol_intervals",
        ]
    if injected_failure_stage_name in (
        "participant_validation",
        "concept_trace_generation",
        "observed_stream_generation",
    ):
        assert observer.operation_calls == [RunExecutionEvent(stage_name="initial_preparation")]
    else:
        failure_call_index = expected_operation_calls.index(expected_failure_event)
        assert observer.operation_calls == expected_operation_calls[: failure_call_index + 1]
    assert observer.finalized_communication_round_counts == []


@pytest.mark.parametrize(
    "injected_failure_stage_name",
    [
        "initial_preparation",
        "sample_processing",
        "started_communication_finalization",
    ],
)
def test_single_run_after_failure_uses_fresh_participants_and_reproduces_success(
    valid_stream_protocol_execution_settings,
    injected_failure_stage_name,
):
    """故障解除後に同factoryを再利用しても参加者と成功結果を新規に得る。"""
    execution_settings = replace(
        valid_stream_protocol_execution_settings,
        experiment_run_conditions=replace(
            valid_stream_protocol_execution_settings.experiment_run_conditions,
            per_client_sample_count=4,
            server_aggregation_interval_per_client_samples=2,
        ),
    )
    first_run_result = runtime_module.execute_stream_protocol_run(
        execution_settings=execution_settings,
        participant_factory=InitialPreparationObservingParticipantFactory(
            observer=RunOperationObserver()
        ),
    )
    observer = RunOperationObserver()
    observer.failure_at_event = RunExecutionEvent(
        stage_name=injected_failure_stage_name,
        client_id=1 if injected_failure_stage_name == "sample_processing" else None,
        sample_index=2 if injected_failure_stage_name == "sample_processing" else None,
        round_index=1 if injected_failure_stage_name == "sample_processing" else None,
    )
    observer.injected_exception = ValueError("再実行前の故障")
    participant_factory = InitialPreparationObservingParticipantFactory(observer=observer)
    with pytest.raises(RunExecutionError):
        runtime_module.execute_stream_protocol_run(
            execution_settings=execution_settings, participant_factory=participant_factory
        )
    first_participants = (
        participant_factory.prepared_participant_history[-1]
        if participant_factory.prepared_participant_history
        else None
    )
    observer.failure_at_event = observer.injected_exception = None
    repeated_run_result = runtime_module.execute_stream_protocol_run(
        execution_settings=execution_settings,
        participant_factory=participant_factory,
    )
    repeated_participants = participant_factory.prepared_participant_history[-1]
    assert first_run_result == repeated_run_result
    assert len(participant_factory.prepared_participant_history) == (
        1 if first_participants is None else 2
    )
    if first_participants is not None:
        assert repeated_participants is not first_participants
        assert repeated_participants.server_operations is not first_participants.server_operations
        for client_id in range(3):
            assert (
                repeated_participants.client_operations[client_id]
                is not first_participants.client_operations[client_id]
            )


def test_single_run_result_cannot_be_changed_through_observer_state(
    valid_stream_protocol_execution_settings,
):
    """処理部・factoryの可変記録を変更しても成功結果は影響を受けない。"""
    observer = RunOperationObserver()
    participant_factory = InitialPreparationObservingParticipantFactory(observer=observer)
    run_result = runtime_module.execute_stream_protocol_run(
        execution_settings=valid_stream_protocol_execution_settings,
        participant_factory=participant_factory,
    )
    observed_client_streams = run_result.observed_client_streams
    evaluation_concept_traces = run_result.evaluation_concept_traces
    expected_execution_events = run_result.execution_events
    observer.operation_calls.clear()
    observer.processed_observed_samples.clear()
    observer.synchronization_readiness_values.clear()
    participant_factory.preparation_observed_samples.clear()
    participant_factory.prepared_participant_history[0].client_operations[0].client_id = 99
    participant_factory.prepared_participant_history.clear()
    participant_factory.borrowed_run_random_sources.python_random_generator.random()
    participant_factory.borrowed_run_random_sources.numpy_random_generator.uniform(size=2)
    assert run_result.observed_client_streams is observed_client_streams
    assert run_result.evaluation_concept_traces is evaluation_concept_traces
    assert run_result.execution_events is expected_execution_events
    assert run_result.observed_client_streams[0].client_id == 0
    assert len(run_result.observed_client_streams[0].observed_samples) == 1500
    assert len(run_result.execution_events) > 4500


@pytest.mark.parametrize(
    "invalid_participant_case",
    [
        "invalid_settings",
        "partial_settings",
        "missing_validate",
        "missing_prepare",
        "noncallable_validate",
        "noncallable_prepare",
        "factory_values",
        "factory_unexpected",
    ],
)
def test_single_run_validation_precedes_random_generation_and_preparation(
    monkeypatch,
    valid_stream_protocol_execution_settings,
    valid_run_settings_mapping,
    invalid_participant_case,
):
    """事前拒否では生成を始めず、factory自身の項目と値も保持する。"""
    participant_factory = ObservingRunParticipantFactory(observer=RunOperationObserver())
    execution_settings = valid_stream_protocol_execution_settings
    invalid_participant_factory = participant_factory
    factory_validation_failure = (
        RunSettingsValidationError(
            configuration_parameter_name="factory_required_condition",
            specified_parameter_value=None,
            validation_failure_reason="接続先の固定条件が不足しています。",
        )
        if invalid_participant_case == "factory_values"
        else ValueError("接続先の事前検査失敗")
    )
    if invalid_participant_case == "invalid_settings":
        execution_settings = None
    elif invalid_participant_case == "partial_settings":
        execution_settings = ValidatedExperimentRunSettingsSubset(**valid_run_settings_mapping)
    elif invalid_participant_case.startswith("missing_"):
        invalid_participant_factory = SimpleNamespace(
            **{
                operation_name: Mock()
                for operation_name in ("validate_configuration", "prepare_run")
                if operation_name
                != (
                    "validate_configuration"
                    if invalid_participant_case == "missing_validate"
                    else "prepare_run"
                )
            }
        )
    elif invalid_participant_case.startswith("noncallable_"):
        setattr(
            participant_factory,
            "validate_configuration"
            if invalid_participant_case.endswith("validate")
            else "prepare_run",
            None,
        )
    else:
        monkeypatch.setattr(
            participant_factory,
            "validate_configuration",
            Mock(side_effect=factory_validation_failure),
        )
    runtime_operation_spies = {
        operation_name: Mock(side_effect=AssertionError("事前検査中に生成してはいけません"))
        for operation_name in (
            "create_run_random_sources",
            "SineSampleGenerator",
            "isolated_cpu_torch_random_state",
            "generate_random_client_concept_traces",
            "build_sine_client_observed_streams",
            "run_stream_protocol_intervals",
        )
    }
    for operation_name, run_operation in runtime_operation_spies.items():
        monkeypatch.setattr(runtime_module, operation_name, run_operation)
    if invalid_participant_case == "factory_unexpected":
        with pytest.raises(RunExecutionError) as exception_info:
            runtime_module.execute_stream_protocol_run(
                execution_settings=execution_settings,
                participant_factory=invalid_participant_factory,
            )
        assert exception_info.value.stage_name == "configuration_validation"
        assert exception_info.value.__cause__ is factory_validation_failure
    else:
        with pytest.raises(RunSettingsValidationError) as exception_info:
            runtime_module.execute_stream_protocol_run(
                execution_settings=execution_settings,
                participant_factory=invalid_participant_factory,
            )
        if invalid_participant_case == "factory_values":
            assert exception_info.value is factory_validation_failure
        else:
            assert exception_info.value.configuration_parameter_name == (
                "execution_settings"
                if invalid_participant_case.endswith("settings")
                else "participant_factory"
            )
            assert exception_info.value.specified_parameter_value is (
                execution_settings
                if invalid_participant_case.endswith("settings")
                else invalid_participant_factory
            )
    for run_operation in runtime_operation_spies.values():
        run_operation.assert_not_called()
    assert participant_factory.prepared_participant_history == []
    assert participant_factory.observer.operation_calls == []


def test_single_run_runtime_prepares_once_and_orders_supply_before_execution(
    monkeypatch,
    valid_stream_protocol_execution_settings,
):
    """同じ借用実体で準備を一度だけ行い、真値を進行へ渡さない。"""
    participant_factory = ObservingRunParticipantFactory(observer=RunOperationObserver())
    runtime_operation_spies = {"order": Mock()}
    for operation_name in (
        "validate_stream_protocol_execution_settings",
        "validate_run_participant_factory_contract",
        "create_run_random_sources",
        "SineSampleGenerator",
        "validate_prepared_run_participants",
        "generate_random_client_concept_traces",
        "build_sine_client_observed_streams",
        "run_stream_protocol_intervals",
        "StreamProtocolRunResult",
    ):
        runtime_operation_spies[operation_name] = Mock(
            wraps=getattr(runtime_module, operation_name)
        )
        runtime_operation_spies["order"].attach_mock(
            runtime_operation_spies[operation_name], operation_name
        )
        monkeypatch.setattr(runtime_module, operation_name, runtime_operation_spies[operation_name])
    for operation_name in ("validate_configuration", "prepare_run"):
        runtime_operation_spies[operation_name] = Mock(
            wraps=getattr(participant_factory, operation_name)
        )
        runtime_operation_spies["order"].attach_mock(
            runtime_operation_spies[operation_name], operation_name
        )
        monkeypatch.setattr(
            participant_factory, operation_name, runtime_operation_spies[operation_name]
        )
    run_result = runtime_module.execute_stream_protocol_run(
        execution_settings=valid_stream_protocol_execution_settings,
        participant_factory=participant_factory,
    )
    runtime_call_order = [
        operation_call[0] for operation_call in runtime_operation_spies["order"].mock_calls
    ]
    assert runtime_call_order == [
        "validate_stream_protocol_execution_settings",
        "validate_run_participant_factory_contract",
        "validate_configuration",
        "create_run_random_sources",
        "SineSampleGenerator",
        "prepare_run",
        "validate_prepared_run_participants",
        "generate_random_client_concept_traces",
        "build_sine_client_observed_streams",
        "run_stream_protocol_intervals",
        "StreamProtocolRunResult",
    ]
    runtime_operation_spies["prepare_run"].assert_called_once_with(
        experiment_run_conditions=valid_stream_protocol_execution_settings.experiment_run_conditions,
        run_random_sources=participant_factory.borrowed_run_random_sources,
        sample_generator=participant_factory.borrowed_sample_generator,
    )
    assert participant_factory.borrowed_sample_generator.numpy_random_generator is (
        participant_factory.borrowed_run_random_sources.numpy_random_generator
    )
    assert (
        runtime_operation_spies["generate_random_client_concept_traces"].call_args.kwargs[
            "python_random_generator"
        ]
        is participant_factory.borrowed_run_random_sources.python_random_generator
    )
    runtime_operation_spies["build_sine_client_observed_streams"].assert_called_once_with(
        evaluation_concept_traces=run_result.evaluation_concept_traces,
        sample_generator=participant_factory.borrowed_sample_generator,
    )
    runtime_operation_spies["run_stream_protocol_intervals"].assert_called_once_with(
        participants=participant_factory.prepared_participant_history[0],
        observed_client_streams=run_result.observed_client_streams,
        server_aggregation_interval_per_client_samples=50,
    )


@pytest.mark.parametrize(
    "invalid_participant_case",
    [
        "wrong_type",
        "mutable_clients",
        "too_few",
        "too_many",
        "negative_id",
        "boolean_id",
        "float_id",
        "duplicate_id",
        "unsorted_ids",
        "missing_id",
        "process_observed_sample",
        "flush_pending_local_updates",
        "has_model_ready_for_server_registration",
        "advance_new_model_upload_wait_after_synchronization",
        "finalize_incomplete_candidate_validation",
        "record_client_states_before_synchronization",
        "synchronize_models",
        "finalize_started_communications",
    ],
)
def test_single_run_runtime_rejects_invalid_prepared_participants_before_supply(
    monkeypatch,
    valid_stream_protocol_execution_settings,
    invalid_participant_case,
):
    """件数・ID・操作の不正をデータ供給前に実行契約違反として報告する。"""
    participant_factory = ObservingRunParticipantFactory(observer=RunOperationObserver())
    invalid_prepared_participants = build_observing_run_participants(
        client_count=3, observer=participant_factory.observer
    )
    if invalid_participant_case == "wrong_type":
        invalid_prepared_participants = SimpleNamespace(
            client_operations=(), server_operations=None
        )
    elif invalid_participant_case == "mutable_clients":
        object.__setattr__(
            invalid_prepared_participants,
            "client_operations",
            list(invalid_prepared_participants.client_operations),
        )
    elif invalid_participant_case == "too_few":
        invalid_prepared_participants = replace(
            invalid_prepared_participants,
            client_operations=invalid_prepared_participants.client_operations[:2],
        )
    elif invalid_participant_case == "too_many":
        invalid_prepared_participants = replace(
            invalid_prepared_participants,
            client_operations=invalid_prepared_participants.client_operations * 2,
        )
    elif invalid_participant_case in ("negative_id", "boolean_id", "float_id", "duplicate_id"):
        invalid_prepared_participants.client_operations[1].client_id = {
            "negative_id": -1,
            "boolean_id": True,
            "float_id": 1.0,
            "duplicate_id": 0,
        }[invalid_participant_case]
    elif invalid_participant_case == "unsorted_ids":
        invalid_prepared_participants = replace(
            invalid_prepared_participants,
            client_operations=tuple(reversed(invalid_prepared_participants.client_operations)),
        )
    elif invalid_participant_case == "missing_id":
        del invalid_prepared_participants.client_operations[1].client_id
    elif hasattr(invalid_prepared_participants.client_operations[1], invalid_participant_case):
        setattr(invalid_prepared_participants.client_operations[1], invalid_participant_case, None)
    else:
        setattr(invalid_prepared_participants.server_operations, invalid_participant_case, None)
    monkeypatch.setattr(
        participant_factory, "prepare_run", Mock(return_value=invalid_prepared_participants)
    )
    runtime_operation_spies = {
        operation_name: Mock(side_effect=AssertionError("参加者検査中に供給してはいけません"))
        for operation_name in (
            "generate_random_client_concept_traces",
            "build_sine_client_observed_streams",
            "run_stream_protocol_intervals",
        )
    }
    for operation_name, run_operation in runtime_operation_spies.items():
        monkeypatch.setattr(runtime_module, operation_name, run_operation)
    with pytest.raises(RunExecutionError) as exception_info:
        runtime_module.execute_stream_protocol_run(
            execution_settings=valid_stream_protocol_execution_settings,
            participant_factory=participant_factory,
        )
    assert exception_info.value.stage_name == "participant_validation"
    assert isinstance(exception_info.value.__cause__, (TypeError, ValueError))
    for run_operation in runtime_operation_spies.values():
        run_operation.assert_not_called()
    assert participant_factory.observer.operation_calls == []


@pytest.mark.parametrize(
    "per_client_sample_count,server_aggregation_interval_per_client_samples",
    [(1500, 50), (16, 7), (3, 7)],
)
def test_single_run_runtime_returns_immutable_observations_truth_counts_and_events(
    valid_stream_protocol_execution_settings,
    per_client_sample_count,
    server_aggregation_interval_per_client_samples,
):
    """生成・処理・末尾件数と不変データを基準・端数・区間長超過で確認する。"""
    observer = RunOperationObserver()
    execution_settings = replace(
        valid_stream_protocol_execution_settings,
        experiment_run_conditions=replace(
            valid_stream_protocol_execution_settings.experiment_run_conditions,
            per_client_sample_count=per_client_sample_count,
            server_aggregation_interval_per_client_samples=server_aggregation_interval_per_client_samples,
        ),
    )
    original_cpu_torch_random_state = torch.get_rng_state().clone()
    run_result = runtime_module.execute_stream_protocol_run(
        execution_settings=execution_settings,
        participant_factory=ObservingRunParticipantFactory(observer=observer),
    )
    assert torch.equal(torch.get_rng_state(), original_cpu_torch_random_state)
    assert run_result.generated_sample_count_per_client == per_client_sample_count
    assert (
        run_result.synchronization_interval_count
        == per_client_sample_count // server_aggregation_interval_per_client_samples
    )
    assert (
        run_result.processed_sample_count_per_client
        == run_result.synchronization_interval_count
        * server_aggregation_interval_per_client_samples
    )
    assert (
        run_result.unprocessed_tail_sample_count_per_client
        == per_client_sample_count % server_aggregation_interval_per_client_samples
    )
    assert (
        len(observer.processed_observed_samples) == 3 * run_result.processed_sample_count_per_client
    )
    for client_id, client_observed_stream in enumerate(run_result.observed_client_streams):
        assert client_observed_stream.client_id == client_id
        assert len(client_observed_stream.observed_samples) == per_client_sample_count
        assert (
            len(run_result.evaluation_concept_traces[client_id].concept_ids_by_sample_index)
            == per_client_sample_count
        )
        assert not hasattr(client_observed_stream.observed_samples[0], "concept_id")
    assert run_result.execution_events[:5] == tuple(
        RunExecutionEvent(stage_name=stage_name)
        for stage_name in (
            "configuration_validation",
            "initial_preparation",
            "participant_validation",
            "concept_trace_generation",
            "observed_stream_generation",
        )
    )
    assert run_result.execution_events[5:] == tuple(observer.operation_calls[1:])
    assert set(run_result.__dataclass_fields__) == {
        "observed_client_streams",
        "evaluation_concept_traces",
        "generated_sample_count_per_client",
        "processed_sample_count_per_client",
        "unprocessed_tail_sample_count_per_client",
        "synchronization_interval_count",
        "execution_events",
    }
    with pytest.raises(FrozenInstanceError):
        run_result.processed_sample_count_per_client = 0


@pytest.mark.parametrize(
    "stage_name,operation_name",
    [
        ("initial_preparation", "prepare_run"),
        ("concept_trace_generation", "generate_random_client_concept_traces"),
        ("observed_stream_generation", "build_sine_client_observed_streams"),
    ],
)
def test_single_run_runtime_reports_preparation_and_supply_failure_stages(
    monkeypatch,
    valid_stream_protocol_execution_settings,
    stage_name,
    operation_name,
):
    """準備・系列・標本の元例外を保持し、失敗後の供給・進行を呼ばない。"""
    participant_factory = ObservingRunParticipantFactory(observer=RunOperationObserver())
    original_exception = ValueError("準備または供給の失敗")
    runtime_operation_spies = {"order": Mock()}
    for record_field_name in (
        "prepare_run",
        "generate_random_client_concept_traces",
        "build_sine_client_observed_streams",
        "run_stream_protocol_intervals",
    ):
        run_operation = getattr(
            participant_factory if record_field_name == "prepare_run" else runtime_module,
            record_field_name,
        )
        runtime_operation_spies[record_field_name] = (
            Mock(side_effect=original_exception)
            if record_field_name == operation_name
            else Mock(wraps=run_operation)
        )
        runtime_operation_spies["order"].attach_mock(
            runtime_operation_spies[record_field_name], record_field_name
        )
        monkeypatch.setattr(
            participant_factory if record_field_name == "prepare_run" else runtime_module,
            record_field_name,
            runtime_operation_spies[record_field_name],
        )
    original_cpu_torch_random_state = torch.get_rng_state().clone()
    with pytest.raises(RunExecutionError) as exception_info:
        runtime_module.execute_stream_protocol_run(
            execution_settings=valid_stream_protocol_execution_settings,
            participant_factory=participant_factory,
        )
    assert exception_info.value.stage_name == stage_name
    assert exception_info.value.__cause__ is original_exception
    assert torch.equal(torch.get_rng_state(), original_cpu_torch_random_state)
    runtime_call_order = [
        operation_call[0] for operation_call in runtime_operation_spies["order"].mock_calls
    ]
    assert (
        runtime_call_order
        == [
            "prepare_run",
            "generate_random_client_concept_traces",
            "build_sine_client_observed_streams",
        ][
            : [
                "prepare_run",
                "generate_random_client_concept_traces",
                "build_sine_client_observed_streams",
            ].index(operation_name)
            + 1
        ]
    )
    runtime_operation_spies["run_stream_protocol_intervals"].assert_not_called()


def test_single_run_runtime_restores_cpu_random_state_when_result_construction_fails(
    monkeypatch,
    valid_stream_protocol_execution_settings,
):
    """成功処理後の結果構築例外にもCPU乱数境界のfinallyを適用する。"""
    observer = RunOperationObserver()
    original_exception = ValueError("結果構築の予期しない失敗")

    def run_operation(**record_field_values):
        # 初期準備と観測処理はtorchを消費しないので、結果構築もrunのseed境界内と確認する。
        assert torch.equal(
            torch.get_rng_state(), torch.Generator(device="cpu").manual_seed(0).get_state()
        )
        torch.rand(1)
        raise original_exception

    monkeypatch.setattr(runtime_module, "StreamProtocolRunResult", Mock(side_effect=run_operation))
    original_cpu_torch_random_state = torch.get_rng_state().clone()
    with pytest.raises(ValueError) as exception_info:
        runtime_module.execute_stream_protocol_run(
            execution_settings=valid_stream_protocol_execution_settings,
            participant_factory=ObservingRunParticipantFactory(observer=observer),
        )
    assert exception_info.value is original_exception
    assert torch.equal(torch.get_rng_state(), original_cpu_torch_random_state)
    assert observer.finalized_communication_round_counts == [30]


def test_single_run_runtime_requires_keyword_arguments(valid_stream_protocol_execution_settings):
    with pytest.raises(TypeError):
        runtime_module.execute_stream_protocol_run(
            valid_stream_protocol_execution_settings,
            ObservingRunParticipantFactory(observer=RunOperationObserver()),
        )


def test_interval_execution_matches_sample_client_and_synchronization_order():
    """3×1500・区間50で4500標本と30同期の全呼出順を照合する。"""
    observer = RunOperationObserver()
    participants = build_observing_run_participants(client_count=3, observer=observer)
    observed_client_streams = build_interval_test_observed_streams(
        client_count=3, per_client_sample_count=1500
    )
    execution_events = run_stream_protocol_intervals(
        participants=participants,
        observed_client_streams=observed_client_streams,
        server_aggregation_interval_per_client_samples=50,
    )
    expected_operation_calls = []
    for round_index in range(30):
        expected_operation_calls.extend(
            RunExecutionEvent(
                stage_name="sample_processing",
                client_id=client_id,
                sample_index=sample_index,
                round_index=round_index,
            )
            for sample_index in range(round_index * 50, (round_index + 1) * 50)
            for client_id in range(3)
        )
        expected_operation_calls.extend(
            RunExecutionEvent(
                stage_name="pending_update_flush",
                client_id=client_id,
                round_index=round_index,
            )
            for client_id in range(3)
        )
        expected_operation_calls.append(
            RunExecutionEvent(stage_name="pre_sync_recording", round_index=round_index)
        )
        expected_operation_calls.extend(
            RunExecutionEvent(
                stage_name="registration_readiness_check",
                client_id=client_id,
                round_index=round_index,
            )
            for client_id in range(3)
        )
        expected_operation_calls.append(
            RunExecutionEvent(stage_name="server_synchronization", round_index=round_index)
        )
        expected_operation_calls.extend(
            RunExecutionEvent(
                stage_name="upload_wait_advance",
                client_id=client_id,
                round_index=round_index,
            )
            for client_id in range(3)
        )
    expected_operation_calls.extend(
        RunExecutionEvent(
            stage_name="incomplete_candidate_validation_finalization",
            client_id=client_id,
        )
        for client_id in range(3)
    )
    expected_operation_calls.append(
        RunExecutionEvent(stage_name="started_communication_finalization")
    )
    assert execution_events == tuple(expected_operation_calls)
    assert observer.operation_calls == expected_operation_calls
    assert len(observer.processed_observed_samples) == 4500
    assert len(observer.synchronization_readiness_values) == 30
    assert observer.finalized_communication_round_counts == [30]
    for client_id, sample_index, observed_sample in observer.processed_observed_samples:
        assert observed_sample is observed_client_streams[client_id].observed_samples[sample_index]


@pytest.mark.parametrize(
    "per_client_sample_count,server_aggregation_interval_per_client_samples", [(16, 7), (3, 7)]
)
def test_interval_execution_processes_only_complete_intervals_and_preserves_terminal_order(
    per_client_sample_count,
    server_aggregation_interval_per_client_samples,
):
    """末尾を処理せず、全client候補終端から通信終端へ進む。"""
    observer = RunOperationObserver()
    participants = build_observing_run_participants(client_count=3, observer=observer)
    observed_client_streams = build_interval_test_observed_streams(
        client_count=3,
        per_client_sample_count=per_client_sample_count,
    )
    execution_events = run_stream_protocol_intervals(
        participants=participants,
        observed_client_streams=observed_client_streams,
        server_aggregation_interval_per_client_samples=server_aggregation_interval_per_client_samples,
    )
    completed_round_count = (
        per_client_sample_count // server_aggregation_interval_per_client_samples
    )
    processed_sample_count_per_client = (
        completed_round_count * server_aggregation_interval_per_client_samples
    )
    assert [
        (client_id, sample_index)
        for client_id, sample_index, observed_sample in observer.processed_observed_samples
    ] == [
        (client_id, sample_index)
        for sample_index in range(processed_sample_count_per_client)
        for client_id in range(3)
    ]
    expected_execution_events = tuple(
        RunExecutionEvent(
            stage_name="incomplete_candidate_validation_finalization",
            client_id=client_id,
        )
        for client_id in range(3)
    ) + (RunExecutionEvent(stage_name="started_communication_finalization"),)
    assert execution_events[-4:] == expected_execution_events
    assert observer.operation_calls[-4:] == list(expected_execution_events)
    assert len(observer.synchronization_readiness_values) == completed_round_count
    assert observer.finalized_communication_round_counts == [completed_round_count]
    if completed_round_count == 0:
        assert execution_events == expected_execution_events


@pytest.mark.parametrize(
    "registration_readiness_by_client,expected_registration_available",
    [
        ({0: True, 1: False, 2: False}, True),
        ({0: False, 1: False, 2: True}, True),
        ({0: False, 1: False, 2: False}, False),
    ],
)
def test_interval_execution_checks_every_client_and_preserves_readiness_at_sync_time(
    registration_readiness_by_client,
    expected_registration_available,
):
    """Trueを得た後も全clientを照会して同期にbool事実を渡す。"""
    observer = RunOperationObserver()
    observer.registration_readiness_by_client = registration_readiness_by_client
    execution_events = run_stream_protocol_intervals(
        participants=build_observing_run_participants(client_count=3, observer=observer),
        observed_client_streams=build_interval_test_observed_streams(
            client_count=3, per_client_sample_count=4
        ),
        server_aggregation_interval_per_client_samples=2,
    )
    assert [
        (operation_call.round_index, operation_call.client_id)
        for operation_call in execution_events
        if operation_call.stage_name == "registration_readiness_check"
    ] == [(round_index, client_id) for round_index in range(2) for client_id in range(3)]
    assert observer.synchronization_readiness_values == [expected_registration_available] * 2
    assert all(
        type(operation_result) is bool
        for operation_result in observer.synchronization_readiness_values
    )


@pytest.mark.parametrize("invalid_readiness_value", [0, 1, None, "True", np.bool_(True)])
def test_interval_execution_rejects_non_boolean_registration_readiness(invalid_readiness_value):
    """非boolをready段階の失敗として報告し、同期や終端を呼ばない。"""
    observer = RunOperationObserver()
    observer.registration_readiness_by_client[1] = invalid_readiness_value
    with pytest.raises(RunExecutionError) as exception_info:
        run_stream_protocol_intervals(
            participants=build_observing_run_participants(client_count=3, observer=observer),
            observed_client_streams=build_interval_test_observed_streams(
                client_count=3, per_client_sample_count=4
            ),
            server_aggregation_interval_per_client_samples=2,
        )
    assert exception_info.value.stage_name == "registration_readiness_check"
    assert exception_info.value.client_id == 1
    assert exception_info.value.round_index == 0
    assert exception_info.value.sample_index is None
    assert isinstance(exception_info.value.__cause__, TypeError)
    assert observer.operation_calls[-1] == RunExecutionEvent(
        stage_name="registration_readiness_check",
        client_id=1,
        round_index=0,
    )
    assert not observer.synchronization_readiness_values
    assert not observer.finalized_communication_round_counts


@pytest.mark.parametrize(
    "expected_failure_event",
    [
        RunExecutionEvent(
            stage_name="sample_processing", client_id=1, sample_index=2, round_index=1
        ),
        RunExecutionEvent(stage_name="pending_update_flush", client_id=1, round_index=1),
        RunExecutionEvent(stage_name="pre_sync_recording", round_index=1),
        RunExecutionEvent(stage_name="registration_readiness_check", client_id=1, round_index=1),
        RunExecutionEvent(stage_name="server_synchronization", round_index=1),
        RunExecutionEvent(stage_name="upload_wait_advance", client_id=1, round_index=1),
        RunExecutionEvent(stage_name="incomplete_candidate_validation_finalization", client_id=1),
        RunExecutionEvent(stage_name="started_communication_finalization"),
    ],
)
def test_interval_execution_reports_failure_positions_and_stops_without_finalization(
    expected_failure_event,
):
    """故障した試行までのprefixだけを呼び、位置・元causeを保持する。"""
    observer = RunOperationObserver()
    observed_client_streams = build_interval_test_observed_streams(
        client_count=3, per_client_sample_count=4
    )
    expected_execution_events = run_stream_protocol_intervals(
        participants=build_observing_run_participants(client_count=3, observer=observer),
        observed_client_streams=observed_client_streams,
        server_aggregation_interval_per_client_samples=2,
    )
    failure_call_index = expected_execution_events.index(expected_failure_event)
    observer = RunOperationObserver()
    original_exception = ValueError("接続先へ注入した故障")
    observer.failure_at_event = expected_failure_event
    observer.injected_exception = original_exception
    with pytest.raises(RunExecutionError) as exception_info:
        run_stream_protocol_intervals(
            participants=build_observing_run_participants(client_count=3, observer=observer),
            observed_client_streams=observed_client_streams,
            server_aggregation_interval_per_client_samples=2,
        )
    assert exception_info.value.__cause__ is original_exception
    assert exception_info.value.stage_name == expected_failure_event.stage_name
    assert exception_info.value.client_id == expected_failure_event.client_id
    assert exception_info.value.sample_index == expected_failure_event.sample_index
    assert exception_info.value.round_index == expected_failure_event.round_index
    assert observer.operation_calls == list(expected_execution_events[: failure_call_index + 1])
    assert observer.finalized_communication_round_counts == []


def test_operation_invocation_records_only_successful_events():
    """操作とstrict bool検証が成功した後だけイベントを追加する。"""
    observer = RunOperationObserver()
    execution_events = []
    run_operation = lambda: observer.record_operation_call(stage_name="initial_preparation")
    assert (
        invoke_run_operation_and_record_success(
            run_operation=run_operation,
            operation_arguments={},
            execution_events=execution_events,
            stage_name="initial_preparation",
        )
        is None
    )
    assert execution_events == observer.operation_calls
    observer.failure_at_event = RunExecutionEvent(stage_name="initial_preparation")
    observer.injected_exception = ValueError("故障")
    with pytest.raises(RunExecutionError):
        invoke_run_operation_and_record_success(
            run_operation=run_operation,
            operation_arguments={},
            execution_events=execution_events,
            stage_name="initial_preparation",
        )
    assert len(execution_events) == 1
    with pytest.raises(RunExecutionError):
        invoke_run_operation_and_record_success(
            run_operation=lambda: 1,
            operation_arguments={},
            execution_events=execution_events,
            stage_name="registration_readiness_check",
            client_id=1,
            round_index=0,
        )
    assert len(execution_events) == 1


def test_interval_execution_requires_keyword_arguments():
    """公開進行関数の参加者・観測列・同期区間をkeywordで渡す。"""
    with pytest.raises(TypeError):
        run_stream_protocol_intervals(
            build_observing_run_participants(client_count=1, observer=RunOperationObserver()),
            build_interval_test_observed_streams(client_count=1, per_client_sample_count=1),
            1,
        )


@pytest.fixture
def valid_run_result_field_values():
    """観測・評価情報・順序を含む独立した成功記録入力を返す。"""
    return {
        "observed_client_streams": (
            ClientObservedStream(
                client_id=0,
                observed_samples=(
                    ObservedSample(feature_values=(0.125, 0.75), class_label=0),
                    ObservedSample(feature_values=(0.5, 0.25), class_label=1),
                ),
            ),
        ),
        "evaluation_concept_traces": (
            ClientConceptTrace(
                client_id=0,
                concept_ids_by_sample_index=(0, 1),
            ),
        ),
        "generated_sample_count_per_client": 2,
        "processed_sample_count_per_client": 2,
        "unprocessed_tail_sample_count_per_client": 0,
        "synchronization_interval_count": 1,
        "execution_events": (
            RunExecutionEvent(
                stage_name="sample_processing", client_id=0, sample_index=0, round_index=0
            ),
            RunExecutionEvent(
                stage_name="sample_processing", client_id=0, sample_index=1, round_index=0
            ),
        ),
    }


@pytest.mark.parametrize(
    "protocol_type,expected_protocol_member_names",
    [
        (RunParticipantFactory, {"validate_configuration", "prepare_run"}),
        (
            RunClientOperations,
            {
                "client_id",
                "process_observed_sample",
                "flush_pending_local_updates",
                "has_model_ready_for_server_registration",
                "advance_new_model_upload_wait_after_synchronization",
                "finalize_incomplete_candidate_validation",
            },
        ),
        (
            RunServerOperations,
            {
                "record_client_states_before_synchronization",
                "synchronize_models",
                "finalize_started_communications",
            },
        ),
    ],
)
def test_run_participant_protocols_expose_only_declared_observation_operations(
    protocol_type,
    expected_protocol_member_names,
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
            "self",
            *operation_parameters[operation_name],
        )
        for position_parameter_name in operation_parameters[operation_name]:
            assert (
                inspect.signature(getattr(protocol_type, operation_name))
                .parameters[position_parameter_name]
                .kind
                is inspect.Parameter.KEYWORD_ONLY
            )
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
        client_operations=(operation_reference,),
        server_operations=operation_reference,
    )
    assert participants.client_operations[0] is operation_reference
    assert participants.server_operations is operation_reference
    for protocol_type in (RunClientOperations, RunServerOperations):
        for operation_name in get_protocol_members(protocol_type) - {"client_id"}:
            assert callable(getattr(operation_reference, operation_name))
            assert (
                tuple(inspect.signature(getattr(operation_reference, operation_name)).parameters)
                == tuple(inspect.signature(getattr(protocol_type, operation_name)).parameters)[1:]
            )
    operation_reference.client_id = 7
    assert participants.client_operations[0].client_id == 7
    for record_field_name in ("client_operations", "server_operations"):
        with pytest.raises(FrozenInstanceError):
            setattr(participants, record_field_name, None)
        with pytest.raises(FrozenInstanceError):
            delattr(participants, record_field_name)
    with pytest.raises(TypeError, match="client_operations"):
        RunParticipants(
            client_operations=[operation_reference], server_operations=operation_reference
        )
    with pytest.raises(TypeError):
        RunParticipants((operation_reference,), operation_reference)


def test_run_execution_events_preserve_stage_and_optional_positions():
    """正式な各stageと、0始まりの位置または対象のないNoneを保持する。"""
    stage_names = (
        "configuration_validation",
        "initial_preparation",
        "participant_validation",
        "concept_trace_generation",
        "observed_stream_generation",
        "sample_processing",
        "pending_update_flush",
        "pre_sync_recording",
        "registration_readiness_check",
        "server_synchronization",
        "upload_wait_advance",
        "incomplete_candidate_validation_finalization",
        "started_communication_finalization",
    )
    for stage_name in stage_names:
        execution_event = RunExecutionEvent(stage_name=stage_name)
        assert execution_event.stage_name == stage_name
        assert (
            execution_event.client_id
            is execution_event.sample_index
            is execution_event.round_index
            is None
        )
    execution_event = RunExecutionEvent(
        stage_name="sample_processing",
        client_id=0,
        sample_index=1499,
        round_index=29,
    )
    assert (
        execution_event.client_id,
        execution_event.sample_index,
        execution_event.round_index,
    ) == (0, 1499, 29)


def test_run_execution_records_are_frozen_and_keyword_only(valid_run_result_field_values):
    """成功結果とイベントの全フィールドの変更・削除・位置引数を拒否する。"""
    for record_type, record_field_values in (
        (
            RunExecutionEvent,
            {
                "stage_name": "sample_processing",
                "client_id": 0,
                "sample_index": 0,
                "round_index": 0,
            },
        ),
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
    assert tuple(
        execution_event.sample_index for execution_event in run_result.execution_events
    ) == (0, 1)
    with pytest.raises(TypeError):
        run_result.execution_events[0] = None
    with pytest.raises(FrozenInstanceError):
        run_result.observed_client_streams[0].observed_samples[0].class_label = 1


@pytest.mark.parametrize(
    "record_field_name,invalid_field_value",
    [
        ("observed_client_streams", []),
        ("observed_client_streams", (SimpleNamespace(),)),
        ("evaluation_concept_traces", []),
        ("evaluation_concept_traces", (SimpleNamespace(),)),
        ("execution_events", []),
        ("execution_events", (SimpleNamespace(),)),
        *[
            (record_field_name, invalid_field_value)
            for record_field_name in (
                "generated_sample_count_per_client",
                "processed_sample_count_per_client",
                "unprocessed_tail_sample_count_per_client",
                "synchronization_interval_count",
            )
            for invalid_field_value in (-1, True, 0.0, "0", None)
        ],
    ],
)
def test_stream_protocol_results_reject_mutable_or_invalid_records(
    valid_run_result_field_values,
    record_field_name,
    invalid_field_value,
):
    """入れ子の不変記録以外や可変集合、不正な件数を保持しない。"""
    valid_run_result_field_values[record_field_name] = invalid_field_value
    with pytest.raises((TypeError, ValueError), match=record_field_name):
        StreamProtocolRunResult(**valid_run_result_field_values)


def test_run_execution_errors_preserve_stage_positions_reason_and_cause():
    """正式stage・位置・理由と、標準causeによる元例外の参照を保持する。"""
    original_exception = ValueError("接続先の失敗")
    run_execution_error = RunExecutionError(
        stage_name="sample_processing",
        client_id=2,
        sample_index=101,
        round_index=2,
        failure_reason="標本処理に失敗しました。",
    )
    with pytest.raises(RunExecutionError) as exception_info:
        raise run_execution_error from original_exception
    assert exception_info.value is run_execution_error
    assert run_execution_error.__cause__ is original_exception
    assert isinstance(run_execution_error, RuntimeError)
    assert run_execution_error.stage_name == "sample_processing"
    assert (
        run_execution_error.client_id,
        run_execution_error.sample_index,
        run_execution_error.round_index,
    ) == (2, 101, 2)
    assert run_execution_error.failure_reason == "標本処理に失敗しました。"
    assert all(
        str(position_parameter_value) in str(run_execution_error)
        for position_parameter_value in ("sample_processing", 2, 101, "標本処理に失敗しました。")
    )
    run_execution_error = RunExecutionError(
        stage_name="initial_preparation", failure_reason="準備に失敗しました。"
    )
    assert (
        run_execution_error.client_id
        is run_execution_error.sample_index
        is run_execution_error.round_index
        is None
    )


@pytest.mark.parametrize("record_type", [RunExecutionEvent, RunExecutionError])
@pytest.mark.parametrize(
    "record_field_name,invalid_field_value",
    [
        ("stage_name", "unknown"),
        ("stage_name", True),
        *[
            (record_field_name, invalid_field_value)
            for record_field_name in ("client_id", "sample_index", "round_index")
            for invalid_field_value in (-1, True, 0.0, "0")
        ],
    ],
)
def test_run_execution_records_reject_unknown_stages_and_invalid_positions(
    record_type,
    record_field_name,
    invalid_field_value,
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
    assert (
        run_random_sources.python_random_generator
        is not repeated_run_random_sources.python_random_generator
    )
    assert (
        run_random_sources.numpy_random_generator
        is not repeated_run_random_sources.numpy_random_generator
    )
    assert (
        run_random_sources.python_random_generator.random()
        == expected_python_random_generator.random()
    )
    np.testing.assert_array_equal(
        run_random_sources.numpy_random_generator.uniform(size=10),
        expected_numpy_random_generator.uniform(size=10),
    )
    assert (
        repeated_run_random_sources.python_random_generator.random() == random.Random(17).random()
    )
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
            dataset_name="sine2",
            random_seed=0,
            client_count=3,
            per_client_sample_count=1500,
            server_aggregation_interval_per_client_samples=50,
        ),
        concept_schedule_settings=RandomConceptScheduleSettings(
            concept_schedule_strategy="random_changes_after_minimum_index_gap",
            minimum_sample_index_gap_before_change_trial=100,
            per_eligible_sample_concept_change_probability=0.015,
        ),
        execution_strategy="sample_index_then_client_order_with_interval_synchronization",
    )


@pytest.mark.parametrize("random_seed", [0, 2**32 - 1])
@pytest.mark.parametrize(
    "per_client_sample_count,server_aggregation_interval_per_client_samples", [(16, 7), (3, 7)]
)
def test_stream_protocol_execution_settings_accept_seed_and_interval_boundaries(
    valid_stream_protocol_execution_settings,
    random_seed,
    per_client_sample_count,
    server_aggregation_interval_per_client_samples,
):
    execution_settings = replace(
        valid_stream_protocol_execution_settings,
        experiment_run_conditions=replace(
            valid_stream_protocol_execution_settings.experiment_run_conditions,
            random_seed=random_seed,
            per_client_sample_count=per_client_sample_count,
            server_aggregation_interval_per_client_samples=server_aggregation_interval_per_client_samples,
        ),
    )
    assert (
        validate_stream_protocol_execution_settings(execution_settings=execution_settings) is None
    )


@pytest.mark.parametrize(
    "configuration_parameter_name,specified_parameter_value",
    [
        ("experiment_run_conditions", None),
        ("concept_schedule_settings", {}),
        ("execution_strategy", "unknown"),
        ("execution_strategy", None),
        (
            "execution_strategy",
            np.array(["sample_index_then_client_order_with_interval_synchronization"]),
        ),
        ("dataset_name", "sea2"),
        ("dataset_name", "mnist2"),
        ("random_seed", 2**32),
    ],
)
def test_stream_protocol_execution_settings_reject_invalid_components_and_choices(
    valid_stream_protocol_execution_settings,
    configuration_parameter_name,
    specified_parameter_value,
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
            replace(
                valid_stream_protocol_execution_settings,
                **{configuration_parameter_name: specified_parameter_value},
            )
    assert exception_info.value.configuration_parameter_name == configuration_parameter_name
    assert exception_info.value.specified_parameter_value == specified_parameter_value
    assert exception_info.value.validation_failure_reason


def test_stream_protocol_execution_validation_rejects_partial_settings(valid_run_settings_mapping):
    partial_run_settings = ValidatedExperimentRunSettingsSubset(**valid_run_settings_mapping)
    with pytest.raises(RunSettingsValidationError) as exception_info:
        validate_stream_protocol_execution_settings(execution_settings=partial_run_settings)
    assert exception_info.value.configuration_parameter_name == "execution_settings"
    assert exception_info.value.specified_parameter_value is partial_run_settings


@pytest.mark.parametrize(
    "missing_parameter_name",
    [
        "experiment_run_conditions",
        "concept_schedule_settings",
        "execution_strategy",
    ],
)
def test_stream_protocol_execution_settings_are_required_frozen_and_keyword_only(
    valid_stream_protocol_execution_settings,
    missing_parameter_name,
):
    with pytest.raises(TypeError):
        StreamProtocolExecutionSettings(
            **{
                configuration_parameter_name: specified_parameter_value
                for configuration_parameter_name, specified_parameter_value in vars(
                    valid_stream_protocol_execution_settings
                ).items()
                if configuration_parameter_name != missing_parameter_name
            }
        )
    with pytest.raises(TypeError):
        StreamProtocolExecutionSettings(*vars(valid_stream_protocol_execution_settings).values())
    with pytest.raises(FrozenInstanceError):
        setattr(valid_stream_protocol_execution_settings, missing_parameter_name, None)
    with pytest.raises(FrozenInstanceError):
        delattr(valid_stream_protocol_execution_settings, missing_parameter_name)
