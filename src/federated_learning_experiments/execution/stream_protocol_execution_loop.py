"""完全な同期区間と終端の操作順を制御する。"""

from collections.abc import Callable

from federated_learning_experiments.data.observed_streams import (
    ClientConceptTrace,
    ClientObservedStream,
)
from federated_learning_experiments.execution.run_execution_errors import RunExecutionError
from federated_learning_experiments.execution.run_execution_records import RunExecutionEvent
from federated_learning_experiments.execution.run_participant_contracts import RunParticipants


def invoke_run_operation_and_record_success(
    *,
    run_operation: Callable[..., object],
    operation_arguments: dict[str, object],
    execution_events: list[RunExecutionEvent],
    stage_name: str,
    client_id: int | None = None,
    sample_index: int | None = None,
    round_index: int | None = None,
) -> object:
    """操作とready検証の成功後に記録し、失敗位置と元causeを保持する。"""
    try:
        operation_result = run_operation(**operation_arguments)
        if stage_name == "registration_readiness_check" and type(operation_result) is not bool:
            raise TypeError("登録可能モデルの照会はboolを返してください。")
    except Exception as caught_exception:
        raise RunExecutionError(
            stage_name=stage_name,
            client_id=client_id,
            sample_index=sample_index,
            round_index=round_index,
            failure_reason=f"{stage_name}の操作に失敗しました。",
        ) from caught_exception
    execution_events.append(
        RunExecutionEvent(
            stage_name=stage_name,
            client_id=client_id,
            sample_index=sample_index,
            round_index=round_index,
        )
    )
    return operation_result


def validate_evaluation_concept_traces_match_observed_streams(
    *,
    evaluation_concept_traces: tuple[ClientConceptTrace, ...],
    observed_client_streams: tuple[ClientObservedStream, ...],
) -> None:
    """概念列が、観測列と、同じclientの順・同じ標本の数で対応することを確かめる。"""
    if type(evaluation_concept_traces) is not tuple:
        raise TypeError("evaluation_concept_tracesには概念列のtupleを指定してください。")
    for evaluation_concept_trace in evaluation_concept_traces:
        if type(evaluation_concept_trace) is not ClientConceptTrace:
            raise TypeError(
                "evaluation_concept_tracesの各要素にはClientConceptTraceを指定してください。"
            )
    if len(evaluation_concept_traces) != len(observed_client_streams):
        raise ValueError("evaluation_concept_tracesの数を、観測列の数と同じにしてください。")
    for evaluation_concept_trace, observed_client_stream in zip(
        evaluation_concept_traces, observed_client_streams, strict=True
    ):
        if evaluation_concept_trace.client_id != observed_client_stream.client_id:
            raise ValueError(
                "evaluation_concept_tracesのclientの順を、観測列と同じにしてください。"
            )
        if len(evaluation_concept_trace.concept_ids_by_sample_index) != len(
            observed_client_stream.observed_samples
        ):
            raise ValueError("evaluation_concept_tracesの標本の数を、観測列と同じにしてください。")


def run_stream_protocol_intervals(
    *,
    participants: RunParticipants,
    observed_client_streams: tuple[ClientObservedStream, ...],
    evaluation_concept_traces: tuple[ClientConceptTrace, ...],
    server_aggregation_interval_per_client_samples: int,
) -> tuple[RunExecutionEvent, ...]:
    """検証済みの参加者と観測列を標本位置・client順に進める。

    概念列（評価用の真値）は、標本位置の値を、診断専用の引数として、clientへ渡すためだけに使う。
    """
    validate_evaluation_concept_traces_match_observed_streams(
        evaluation_concept_traces=evaluation_concept_traces,
        observed_client_streams=observed_client_streams,
    )
    execution_events: list[RunExecutionEvent] = []
    stream_sample_count = len(observed_client_streams[0].observed_samples)
    synchronization_interval_count = (
        stream_sample_count // server_aggregation_interval_per_client_samples
    )
    for round_index in range(synchronization_interval_count):
        interval_start_sample_index = round_index * server_aggregation_interval_per_client_samples
        interval_end_sample_index = (
            interval_start_sample_index + server_aggregation_interval_per_client_samples
        )
        for sample_index in range(interval_start_sample_index, interval_end_sample_index):
            for client_operation, observed_client_stream, evaluation_concept_trace in zip(
                participants.client_operations,
                observed_client_streams,
                evaluation_concept_traces,
                strict=True,
            ):
                invoke_run_operation_and_record_success(
                    run_operation=client_operation.process_observed_sample,
                    operation_arguments={
                        "observed_sample": observed_client_stream.observed_samples[sample_index],
                        "sample_index": sample_index,
                        "evaluation_concept_id": (
                            evaluation_concept_trace.concept_ids_by_sample_index[sample_index]
                        ),
                    },
                    execution_events=execution_events,
                    stage_name="sample_processing",
                    client_id=client_operation.client_id,
                    sample_index=sample_index,
                    round_index=round_index,
                )
        for client_operation in participants.client_operations:
            invoke_run_operation_and_record_success(
                run_operation=client_operation.flush_pending_local_updates,
                operation_arguments={"round_index": round_index},
                execution_events=execution_events,
                stage_name="pending_update_flush",
                client_id=client_operation.client_id,
                round_index=round_index,
            )
        invoke_run_operation_and_record_success(
            run_operation=participants.server_operations.record_client_states_before_synchronization,
            operation_arguments={"round_index": round_index},
            execution_events=execution_events,
            stage_name="pre_sync_recording",
            round_index=round_index,
        )
        registration_readiness_values = []
        for client_operation in participants.client_operations:
            registration_readiness_values.append(
                invoke_run_operation_and_record_success(
                    run_operation=client_operation.has_model_ready_for_server_registration,
                    operation_arguments={},
                    execution_events=execution_events,
                    stage_name="registration_readiness_check",
                    client_id=client_operation.client_id,
                    round_index=round_index,
                )
            )
        new_model_registration_available = any(registration_readiness_values)
        invoke_run_operation_and_record_success(
            run_operation=participants.server_operations.synchronize_models,
            operation_arguments={
                "round_index": round_index,
                "new_model_registration_available": new_model_registration_available,
            },
            execution_events=execution_events,
            stage_name="server_synchronization",
            round_index=round_index,
        )
        for client_operation in participants.client_operations:
            invoke_run_operation_and_record_success(
                run_operation=client_operation.advance_new_model_upload_wait_after_synchronization,
                operation_arguments={"round_index": round_index},
                execution_events=execution_events,
                stage_name="upload_wait_advance",
                client_id=client_operation.client_id,
                round_index=round_index,
            )
    for client_operation in participants.client_operations:
        invoke_run_operation_and_record_success(
            run_operation=client_operation.finalize_incomplete_candidate_validation,
            operation_arguments={},
            execution_events=execution_events,
            stage_name="incomplete_candidate_validation_finalization",
            client_id=client_operation.client_id,
        )
    invoke_run_operation_and_record_success(
        run_operation=participants.server_operations.finalize_started_communications,
        operation_arguments={"completed_round_count": synchronization_interval_count},
        execution_events=execution_events,
        stage_name="started_communication_finalization",
    )
    return tuple(execution_events)
