"""固定条件・初期準備・観測標本の供給・区間進行を単一runへ組み立てる。"""

from typing import cast

from federated_learning_experiments.core.configuration_errors import RunSettingsValidationError
from federated_learning_experiments.data.concept_schedules.random_concept_schedule_generation import (
    generate_random_client_concept_traces,
)
from federated_learning_experiments.data.observed_sample_generation import (
    build_client_observed_streams,
    create_observed_sample_generator,
)
from federated_learning_experiments.data.observed_streams import (
    ClientConceptTrace,
    ClientObservedStream,
)
from federated_learning_experiments.execution.run_execution_errors import RunExecutionError
from federated_learning_experiments.execution.run_execution_records import (
    RunExecutionEvent,
    StreamProtocolRunResult,
)
from federated_learning_experiments.execution.run_participant_contracts import (
    RunParticipantFactory,
    RunParticipants,
)
from federated_learning_experiments.execution.run_random_sources import create_run_random_sources
from federated_learning_experiments.execution.stream_protocol_execution_loop import (
    invoke_run_operation_and_record_success,
    run_stream_protocol_intervals,
)
from federated_learning_experiments.execution.stream_protocol_execution_settings import (
    StreamProtocolExecutionSettings,
    validate_stream_protocol_execution_settings,
)
from federated_learning_experiments.learning.models.torch_random_state_scope import (
    isolated_cpu_torch_random_state,
)


def validate_run_participant_factory_contract(*, participant_factory: object) -> None:
    """初期準備前にfactoryの事前検証と準備操作が存在することを確認する。"""
    for operation_name in ("validate_configuration", "prepare_run"):
        if not callable(getattr(participant_factory, operation_name, None)):
            raise RunSettingsValidationError(
                configuration_parameter_name="participant_factory",
                specified_parameter_value=participant_factory,
                validation_failure_reason=f"{operation_name}を呼び出せるfactoryを指定してください。",
            )


def validate_prepared_run_participants(*, participants: object, client_count: int) -> None:
    """参加者の型・件数・ID・操作を、処理部を呼ばずに検査する。"""
    if type(participants) is not RunParticipants:
        raise TypeError("初期準備はRunParticipantsを返してください。")
    if type(participants.client_operations) is not tuple:
        raise TypeError("client_operationsにはclient操作のtupleを指定してください。")
    if len(participants.client_operations) != client_count:
        raise ValueError(
            f"client_operationsの件数はclient_count={client_count}と一致させてください。"
        )
    required_client_operation_names = (
        "process_observed_sample",
        "flush_pending_local_updates",
        "has_model_ready_for_server_registration",
        "advance_new_model_upload_wait_after_synchronization",
        "finalize_incomplete_candidate_validation",
    )
    for expected_client_id, client_operations_instance in enumerate(participants.client_operations):
        if type(getattr(client_operations_instance, "client_id", None)) is not int:
            raise TypeError(
                f"位置{expected_client_id}のclient_idにはboolを除く整数を指定してください。"
            )
        if client_operations_instance.client_id != expected_client_id:
            raise ValueError(
                f"client_idは0から昇順で重複なく配置してください。期待ID={expected_client_id}"
            )
        for operation_name in required_client_operation_names:
            if not callable(getattr(client_operations_instance, operation_name, None)):
                raise TypeError(
                    f"client_id={expected_client_id}の{operation_name}を呼び出せません。"
                )
    required_server_operation_names = (
        "record_client_states_before_synchronization",
        "synchronize_models",
        "finalize_started_communications",
    )
    for operation_name in required_server_operation_names:
        if not callable(getattr(participants.server_operations, operation_name, None)):
            raise TypeError(f"serverの{operation_name}を呼び出せません。")


def execute_stream_protocol_run(
    *,
    execution_settings: StreamProtocolExecutionSettings,
    participant_factory: RunParticipantFactory,
) -> StreamProtocolRunResult:
    """準備後の乱数を継続し、観測値だけを進行へ渡して不変記録を返す。"""
    validate_stream_protocol_execution_settings(execution_settings=execution_settings)
    try:
        validate_run_participant_factory_contract(participant_factory=participant_factory)
        participant_factory.validate_configuration()
    except RunSettingsValidationError:
        raise
    except Exception as caught_exception:
        raise RunExecutionError(
            stage_name="configuration_validation",
            failure_reason="factoryの事前条件検査に失敗しました。",
        ) from caught_exception
    experiment_run_conditions = execution_settings.experiment_run_conditions
    run_random_sources = create_run_random_sources(
        random_seed=experiment_run_conditions.random_seed
    )
    with isolated_cpu_torch_random_state(random_seed=experiment_run_conditions.random_seed):
        sample_generator = create_observed_sample_generator(
            dataset_name=experiment_run_conditions.dataset_name,
            numpy_random_generator=run_random_sources.numpy_random_generator,
        )
        execution_events = [RunExecutionEvent(stage_name="configuration_validation")]
        # 操作wrapperのobject戻り値を、操作の宣言型へ対応付ける。値は変更しない。
        participants = cast(
            RunParticipants,
            invoke_run_operation_and_record_success(
                run_operation=participant_factory.prepare_run,
                operation_arguments={
                    "experiment_run_conditions": experiment_run_conditions,
                    "run_random_sources": run_random_sources,
                    "sample_generator": sample_generator,
                },
                execution_events=execution_events,
                stage_name="initial_preparation",
            ),
        )
        invoke_run_operation_and_record_success(
            run_operation=validate_prepared_run_participants,
            operation_arguments={
                "participants": participants,
                "client_count": experiment_run_conditions.client_count,
            },
            execution_events=execution_events,
            stage_name="participant_validation",
        )
        evaluation_concept_traces = cast(
            tuple[ClientConceptTrace, ...],
            invoke_run_operation_and_record_success(
                run_operation=generate_random_client_concept_traces,
                operation_arguments={
                    "experiment_run_conditions": experiment_run_conditions,
                    "concept_schedule_settings": execution_settings.concept_schedule_settings,
                    "python_random_generator": run_random_sources.python_random_generator,
                },
                execution_events=execution_events,
                stage_name="concept_trace_generation",
            ),
        )
        observed_client_streams = cast(
            tuple[ClientObservedStream, ...],
            invoke_run_operation_and_record_success(
                run_operation=build_client_observed_streams,
                operation_arguments={
                    "evaluation_concept_traces": evaluation_concept_traces,
                    "sample_generator": sample_generator,
                },
                execution_events=execution_events,
                stage_name="observed_stream_generation",
            ),
        )
        execution_events.extend(
            run_stream_protocol_intervals(
                participants=participants,
                observed_client_streams=observed_client_streams,
                evaluation_concept_traces=evaluation_concept_traces,
                server_aggregation_interval_per_client_samples=(
                    experiment_run_conditions.server_aggregation_interval_per_client_samples
                ),
            )
        )
        synchronization_interval_count = (
            experiment_run_conditions.per_client_sample_count
            // experiment_run_conditions.server_aggregation_interval_per_client_samples
        )
        return StreamProtocolRunResult(
            observed_client_streams=observed_client_streams,
            evaluation_concept_traces=evaluation_concept_traces,
            generated_sample_count_per_client=experiment_run_conditions.per_client_sample_count,
            processed_sample_count_per_client=(
                synchronization_interval_count
                * experiment_run_conditions.server_aggregation_interval_per_client_samples
            ),
            unprocessed_tail_sample_count_per_client=(
                experiment_run_conditions.per_client_sample_count
                % experiment_run_conditions.server_aggregation_interval_per_client_samples
            ),
            synchronization_interval_count=synchronization_interval_count,
            execution_events=tuple(execution_events),
        )
