"""FedDriftの全体runを接続する。"""

from dataclasses import dataclass
from typing import cast

from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.data.concept_schedules.random_concept_schedule_generation import (
    generate_random_client_concept_traces,
)
from federated_learning_experiments.data.concept_schedules.random_concept_schedule_settings import (
    RandomConceptScheduleSettings,
)
from federated_learning_experiments.data.observed_sample_generation import (
    ObservedSampleGenerator,
    build_client_observed_streams,
    create_observed_sample_generator,
)
from federated_learning_experiments.data.observed_streams import (
    ClientConceptTrace,
    ClientObservedStream,
)
from federated_learning_experiments.execution.run_execution_records import (
    RunExecutionEvent,
    StreamProtocolRunResult,
)
from federated_learning_experiments.execution.run_random_sources import (
    RunRandomSources,
    create_run_random_sources,
)
from federated_learning_experiments.execution.stream_protocol_execution_loop import (
    invoke_run_operation_and_record_success,
)
from federated_learning_experiments.learning.models.torch_random_state_scope import (
    isolated_cpu_torch_random_state,
)
from federated_learning_experiments.runtime.feddrift_run_participant_factory import (
    FeddriftRunParticipantFactory,
    FeddriftRunParticipants,
)


@dataclass(frozen=True, kw_only=True)
class FeddriftStreamProtocolExecutionSettings:
    """全体runの条件。"""

    experiment_run_conditions: ExperimentRunConditions
    concept_schedule_settings: RandomConceptScheduleSettings

    def __post_init__(self) -> None:
        for settings, expected in (
            (self.experiment_run_conditions, ExperimentRunConditions),
            (self.concept_schedule_settings, RandomConceptScheduleSettings),
        ):
            if type(settings) is not expected:
                raise TypeError(f"exact {expected.__name__}が必要です。")
            settings.__post_init__()
        if self.experiment_run_conditions.random_seed > 2**32 - 1:
            raise ValueError("seedは2**32-1以下が必要です。")


def _validate_configuration(
    *,
    execution_settings: FeddriftStreamProtocolExecutionSettings,
    participant_factory: FeddriftRunParticipantFactory,
) -> None:
    if type(execution_settings) is not FeddriftStreamProtocolExecutionSettings:
        raise TypeError("exact FeddriftStreamProtocolExecutionSettingsが必要です。")
    execution_settings.__post_init__()
    if type(participant_factory) is not FeddriftRunParticipantFactory:
        raise TypeError("exact FeddriftRunParticipantFactoryが必要です。")
    participant_factory.validate_run_conditions(
        experiment_run_conditions=execution_settings.experiment_run_conditions
    )


def _validate_participants(*, participants: FeddriftRunParticipants, client_count: int) -> None:
    if type(participants) is not FeddriftRunParticipants:
        raise TypeError("exact FeddriftRunParticipantsが必要です。")
    participants.__post_init__()
    if len(participants.run_clients) != client_count:
        raise ValueError("準備済みclient数とrun条件を一致させてください。")


def _prepare_participants_and_generator(
    *,
    participant_factory: FeddriftRunParticipantFactory,
    experiment_run_conditions: ExperimentRunConditions,
    run_random_sources: RunRandomSources,
) -> tuple[FeddriftRunParticipants, ObservedSampleGenerator]:
    # MNIST読込みを含む生成器の構築も、初期準備の成功境界に含める。
    generator = create_observed_sample_generator(
        dataset_name=experiment_run_conditions.dataset_name,
        numpy_random_generator=run_random_sources.numpy_random_generator,
    )
    participants = participant_factory.prepare_run(
        experiment_run_conditions=experiment_run_conditions,
        run_random_sources=run_random_sources,
        sample_generator=generator,
    )
    return participants, generator


def execute_feddrift_stream_protocol_run(
    *,
    execution_settings: FeddriftStreamProtocolExecutionSettings,
    participant_factory: FeddriftRunParticipantFactory,
) -> StreamProtocolRunResult:
    """準備から終端まで実行する。"""
    events: list[RunExecutionEvent] = []
    invoke_run_operation_and_record_success(
        run_operation=_validate_configuration,
        operation_arguments=dict(
            execution_settings=execution_settings, participant_factory=participant_factory
        ),
        execution_events=events,
        stage_name="configuration_validation",
    )
    conditions = execution_settings.experiment_run_conditions
    sources = create_run_random_sources(random_seed=conditions.random_seed)
    with isolated_cpu_torch_random_state(random_seed=conditions.random_seed):
        participants, generator = cast(
            tuple[FeddriftRunParticipants, ObservedSampleGenerator],
            invoke_run_operation_and_record_success(
                run_operation=_prepare_participants_and_generator,
                operation_arguments=dict(
                    experiment_run_conditions=conditions,
                    run_random_sources=sources,
                    participant_factory=participant_factory,
                ),
                execution_events=events,
                stage_name="initial_preparation",
            ),
        )
        invoke_run_operation_and_record_success(
            run_operation=_validate_participants,
            operation_arguments=dict(
                participants=participants, client_count=conditions.client_count
            ),
            execution_events=events,
            stage_name="participant_validation",
        )
        traces = cast(
            tuple[ClientConceptTrace, ...],
            invoke_run_operation_and_record_success(
                run_operation=generate_random_client_concept_traces,
                operation_arguments=dict(
                    experiment_run_conditions=conditions,
                    concept_schedule_settings=execution_settings.concept_schedule_settings,
                    python_random_generator=sources.python_random_generator,
                ),
                execution_events=events,
                stage_name="concept_trace_generation",
            ),
        )
        streams = cast(
            tuple[ClientObservedStream, ...],
            invoke_run_operation_and_record_success(
                run_operation=build_client_observed_streams,
                operation_arguments=dict(
                    evaluation_concept_traces=traces, sample_generator=generator
                ),
                execution_events=events,
                stage_name="observed_stream_generation",
            ),
        )
        chunk = conditions.server_aggregation_interval_per_client_samples
        count = conditions.per_client_sample_count // chunk
        for timestep in range(count):
            start = timestep * chunk
            for client, stream, trace in zip(
                participants.run_clients, streams, traces, strict=True
            ):
                invoke_run_operation_and_record_success(
                    run_operation=client.process_observed_block,
                    operation_arguments=dict(
                        observed_samples=stream.observed_samples[start : start + chunk],
                        evaluation_concept_ids=trace.concept_ids_by_sample_index[
                            start : start + chunk
                        ],
                    ),
                    execution_events=events,
                    stage_name="sample_processing",
                    client_id=client.client_id,
                    sample_index=start,
                    round_index=timestep,
                )
            invoke_run_operation_and_record_success(
                run_operation=participants.run_server.synchronize_timestep,
                operation_arguments=dict(timestep_index=timestep),
                execution_events=events,
                stage_name="server_synchronization",
                round_index=timestep,
            )
        invoke_run_operation_and_record_success(
            run_operation=participants.run_server.finalize_observed_stream,
            operation_arguments=dict(timestep_index=max(0, count - 1)),
            execution_events=events,
            stage_name="observed_stream_finalization",
            round_index=max(0, count - 1),
        )
        return StreamProtocolRunResult(
            observed_client_streams=streams,
            evaluation_concept_traces=traces,
            generated_sample_count_per_client=conditions.per_client_sample_count,
            processed_sample_count_per_client=count * chunk,
            unprocessed_tail_sample_count_per_client=conditions.per_client_sample_count % chunk,
            synchronization_interval_count=count,
            execution_events=tuple(events),
        )
