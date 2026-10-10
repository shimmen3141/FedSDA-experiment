"""FedSDAの全体run 1回の結果と参加者の記録から、指標を導出する（状態は変えない）。"""

from dataclasses import dataclass

from torch import Tensor

from federated_learning_experiments.evaluation.communication_volume_record_store import (
    CommunicationVolumeSnapshot,
)
from federated_learning_experiments.evaluation.run_metric_calculations import (
    DetectionMetrics,
    RunMetricSettings,
    calculate_detection_metrics,
    calculate_prediction_accuracy,
    calculate_stable_period_prediction_accuracy,
    extract_concept_change_sample_indices,
)
from federated_learning_experiments.execution.run_execution_records import StreamProtocolRunResult
from federated_learning_experiments.execution.run_participant_contracts import RunParticipants
from federated_learning_experiments.methods.fedsda.model_registration.global_model_repository import (
    GlobalModelRepository,
)
from federated_learning_experiments.runtime.fedsda_run_client import FedsdaRunClient
from federated_learning_experiments.runtime.fedsda_run_server import FedsdaRunServer
from federated_learning_experiments.runtime.server_model_registration_and_aggregation import (
    split_shared_and_concept_specific_parameters,
)


@dataclass(frozen=True, kw_only=True)
class FedsdaRunMetrics:
    """FedSDAの全体run 1回の指標。

    検出の指標は、学習帰属の切替を検出として数える。最終のパラメータ量は、共有部を1回、
    概念固有部をグローバルモデルごとに数える。再較正で再生した標本の数は、全clientの合計。
    """

    prediction_accuracy: float
    stable_period_prediction_accuracy: float
    training_model_switch_detection_metrics: DetectionMetrics
    final_global_model_count: int
    communication_volume: CommunicationVolumeSnapshot
    final_parameter_value_count: int
    final_parameter_byte_count: int
    candidate_validation_decision_count: int
    mixed_prediction_sample_count: int
    prediction_weight_recalibration_replayed_sample_count: int
    global_diagnostic_recalibration_replayed_sample_count: int


def _count_parameter_values_and_bytes(*, parameters: dict[str, Tensor]) -> tuple[int, int]:
    return (
        sum(parameter_values.numel() for parameter_values in parameters.values()),
        sum(
            parameter_values.numel() * parameter_values.element_size()
            for parameter_values in parameters.values()
        ),
    )


def _count_final_parameter_values_and_bytes(
    *, global_model_repository: GlobalModelRepository
) -> tuple[int, int]:
    """共有部は、最初のグローバルモデルのものを1回、概念固有部は、モデルごとに数える。"""
    global_model_ids = global_model_repository.global_model_ids
    if not global_model_ids:
        return 0, 0
    shared_parameters, _ = split_shared_and_concept_specific_parameters(
        parameter_snapshot=global_model_repository.get_global_model_parameters(
            model_id=global_model_ids[0]
        )
    )
    parameter_value_count, parameter_byte_count = _count_parameter_values_and_bytes(
        parameters=shared_parameters
    )
    for model_id in global_model_ids:
        _, concept_specific_parameters = split_shared_and_concept_specific_parameters(
            parameter_snapshot=global_model_repository.get_global_model_parameters(
                model_id=model_id
            )
        )
        concept_specific_value_count, concept_specific_byte_count = (
            _count_parameter_values_and_bytes(parameters=concept_specific_parameters)
        )
        parameter_value_count += concept_specific_value_count
        parameter_byte_count += concept_specific_byte_count
    return parameter_value_count, parameter_byte_count


def derive_fedsda_run_metrics(
    *,
    run_result: StreamProtocolRunResult,
    participants: RunParticipants,
    run_metric_settings: RunMetricSettings,
) -> FedsdaRunMetrics:
    """全体runの結果と参加者から、指標を導出する。

    概念の変更位置は、概念列の全体（処理されなかった末尾を含む）から取る。末尾にある変更は、
    検出されないので、見逃しとして数える。
    """
    if type(run_result) is not StreamProtocolRunResult:
        raise TypeError("run_result must be exact StreamProtocolRunResult")
    if type(participants) is not RunParticipants:
        raise TypeError("participants must be exact RunParticipants")
    if type(run_metric_settings) is not RunMetricSettings:
        raise TypeError("run_metric_settings must be exact RunMetricSettings")
    # 手で壊した設定も、値を使う前に検査し直す。
    run_metric_settings.__post_init__()
    run_clients: list[FedsdaRunClient] = []
    for client_operations in participants.client_operations:
        if type(client_operations) is not FedsdaRunClient:
            raise TypeError("client operations must be exact FedsdaRunClient")
        run_clients.append(client_operations)
    run_server = participants.server_operations
    if type(run_server) is not FedsdaRunServer:
        raise TypeError("server operations must be exact FedsdaRunServer")
    evaluation_concept_traces = run_result.evaluation_concept_traces
    if len(run_clients) != len(evaluation_concept_traces):
        raise ValueError("client count must equal the evaluation concept trace count")
    for run_client, evaluation_concept_trace in zip(
        run_clients, evaluation_concept_traces, strict=True
    ):
        if run_client.client_id != evaluation_concept_trace.client_id:
            raise ValueError("clients must be in the order of the evaluation concept traces")

    prediction_correctness_by_client = tuple(
        tuple(
            prediction_record.combined_prediction_is_correct
            for prediction_record in run_client.owners.sample_prediction_record_store.snapshot_sample_prediction_records()
        )
        for run_client in run_clients
    )
    concept_change_sample_indices_by_client = tuple(
        extract_concept_change_sample_indices(
            concept_ids_by_sample_index=evaluation_concept_trace.concept_ids_by_sample_index
        )
        for evaluation_concept_trace in evaluation_concept_traces
    )
    training_model_switch_sample_indices_by_client = tuple(
        run_client.owners.adaptation_record_store.get_state_snapshot().training_model_switch_sample_indices
        for run_client in run_clients
    )
    server_owners = run_server.owners
    final_parameter_value_count, final_parameter_byte_count = (
        _count_final_parameter_values_and_bytes(
            global_model_repository=server_owners.global_model_repository
        )
    )
    return FedsdaRunMetrics(
        prediction_accuracy=calculate_prediction_accuracy(
            prediction_correctness_by_client=prediction_correctness_by_client
        ),
        stable_period_prediction_accuracy=calculate_stable_period_prediction_accuracy(
            prediction_correctness_by_client=prediction_correctness_by_client,
            concept_change_sample_indices_by_client=concept_change_sample_indices_by_client,
            recovery_window_sample_count=run_metric_settings.post_change_recovery_window_sample_count,
        ),
        training_model_switch_detection_metrics=calculate_detection_metrics(
            concept_change_sample_indices_by_client=concept_change_sample_indices_by_client,
            detection_sample_indices_by_client=training_model_switch_sample_indices_by_client,
            maximum_delay_sample_count=run_metric_settings.maximum_detection_delay_sample_count,
        ),
        final_global_model_count=len(server_owners.global_model_repository.global_model_ids),
        communication_volume=server_owners.communication_volume_record_store.get_state_snapshot(),
        final_parameter_value_count=final_parameter_value_count,
        final_parameter_byte_count=final_parameter_byte_count,
        candidate_validation_decision_count=sum(
            len(
                run_client.owners.candidate_validation_decision_record_store.snapshot_candidate_validation_decision_records()
            )
            for run_client in run_clients
        ),
        # 最終構成は、混合予測が常時有効なので、予測した標本のすべてが、混合予測。
        mixed_prediction_sample_count=sum(
            len(prediction_correctness)
            for prediction_correctness in prediction_correctness_by_client
        ),
        prediction_weight_recalibration_replayed_sample_count=sum(
            run_client.owners.fixed_share_prediction_weight_controller.aggregation_recalibration_sample_count
            for run_client in run_clients
        ),
        global_diagnostic_recalibration_replayed_sample_count=sum(
            run_client.owners.diagnostic_evidence_collection.global_diagnostic_evidence.aggregation_recalibration_sample_count
            for run_client in run_clients
        ),
    )
