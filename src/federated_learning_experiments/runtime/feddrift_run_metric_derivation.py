"""FedDriftの基本指標を共通計算へ接続する。"""

from dataclasses import dataclass

from federated_learning_experiments.evaluation.communication_volume_record_store import (
    CommunicationVolumeSnapshot,
)
from federated_learning_experiments.evaluation.computation_cost_summary import (
    ComputationCostSummary,
    ServerComputationCounts,
    summarize_computation_cost,
)
from federated_learning_experiments.evaluation.held_model_count_record_store import (
    HeldModelCountRecordStore,
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
from federated_learning_experiments.learning.models.model_computation_measurement import (
    ModelComputationCounts,
)
from federated_learning_experiments.runtime.feddrift_client_stream_processing import (
    FeddriftSamplePrediction,
)
from federated_learning_experiments.runtime.feddrift_detection_batch_processing import (
    FeddriftDetectionBatchTransition,
)
from federated_learning_experiments.runtime.feddrift_model_diagnostic_metric_derivation import (
    FeddriftModelDiagnosticMetrics,
    derive_feddrift_model_diagnostic_metrics,
)
from federated_learning_experiments.runtime.feddrift_run_participant_factory import (
    FeddriftRunParticipants,
)


@dataclass(frozen=True, kw_only=True)
class FeddriftRunMetrics:
    """共通定義によるFedDrift基本指標。"""

    model_diagnostic_metrics: FeddriftModelDiagnosticMetrics
    prediction_accuracy: float
    stable_period_prediction_accuracy: float
    training_model_switch_detection_metrics: DetectionMetrics
    final_global_model_count: int
    communication_volume: CommunicationVolumeSnapshot
    final_parameter_value_count: int
    final_parameter_byte_count: int
    model_computation_counts: ModelComputationCounts | None
    server_computation_counts: ServerComputationCounts
    processed_sample_count: int
    held_model_sample_count: int
    client_consolidation_parameter_multiply_accumulate_count: int
    computation_cost_summary: ComputationCostSummary | None


def derive_feddrift_run_metrics(
    *,
    run_result: StreamProtocolRunResult,
    participants: FeddriftRunParticipants,
    run_metric_settings: RunMetricSettings,
    model_computation_counts: ModelComputationCounts | None = None,
) -> FeddriftRunMetrics:
    """記録の対応を検査してから指標を導出する。"""
    for value, expected in (
        (run_result, StreamProtocolRunResult),
        (participants, FeddriftRunParticipants),
        (run_metric_settings, RunMetricSettings),
    ):
        if type(value) is not expected:
            raise TypeError(f"exact {expected.__name__}が必要です。")
        value.__post_init__()
    if not (
        len(participants.run_clients)
        == len(run_result.observed_client_streams)
        == len(run_result.evaluation_concept_traces)
    ):
        raise ValueError("client/観測/概念の件数を一致させてください。")
    generated = run_result.generated_sample_count_per_client
    processed = run_result.processed_sample_count_per_client
    for event in run_result.execution_events:
        event.__post_init__()
    held_model_sample_count = 0
    for client, stream, trace in zip(
        participants.run_clients,
        run_result.observed_client_streams,
        run_result.evaluation_concept_traces,
        strict=True,
    ):
        stream.__post_init__()
        trace.__post_init__()
        client._settings.__post_init__()
        chunk = client._settings.detection_batch_sample_count
        if (
            stream.client_id != client.client_id
            or trace.client_id != client.client_id
            or len(stream.observed_samples) != generated
            or len(trace.concept_ids_by_sample_index) != generated
        ):
            raise ValueError("client IDと生成件数の対応が不正です。")
        if (
            processed != generated // chunk * chunk
            or run_result.unprocessed_tail_sample_count_per_client != generated % chunk
            or run_result.synchronization_interval_count != generated // chunk
        ):
            raise ValueError("生成/処理/末尾/同期件数が対応していません。")
        predictions = client.stream_processor.sample_predictions
        if (
            type(client.stream_processor.processed_sample_count) is not int
            or client.stream_processor.processed_sample_count != processed
            or len(predictions) != processed
        ):
            raise ValueError("予測記録と処理件数が対応していません。")
        store = client.stream_processor.held_model_count_record_store
        if type(store) is not HeldModelCountRecordStore:
            raise TypeError("exact HeldModelCountRecordStoreが必要です。")
        counts = store.snapshot_held_model_counts()
        if len(counts) != processed:
            raise ValueError("保有数記録と処理件数が対応していません。")
        for count in counts:
            if type(count) is not int:
                raise TypeError("保有数はbuiltin intが必要です。")
            if count < 1:
                raise ValueError("保有数は正が必要です。")
        held_model_sample_count += sum(counts)
        for sample in stream.observed_samples:
            sample.__post_init__()
        transitions = client.stream_processor.completed_detection_batch_transitions
        if len(transitions) != run_result.synchronization_interval_count:
            raise ValueError("検出batch記録と同期件数が対応していません。")
        for round_index, transition in enumerate(transitions):
            if type(transition) is not FeddriftDetectionBatchTransition:
                raise TypeError("exact FeddriftDetectionBatchTransitionが必要です。")
            transition.__post_init__()
            if (transition.first_sample_index, transition.last_sample_index) != (
                round_index * chunk,
                (round_index + 1) * chunk - 1,
            ):
                raise ValueError("検出batch位置が観測位置と対応していません。")
        for index, prediction in enumerate(predictions):
            if type(prediction) is not FeddriftSamplePrediction:
                raise TypeError("exact FeddriftSamplePredictionが必要です。")
            prediction.__post_init__()
            if (
                prediction.sample_index != index
                or prediction.observed_class_id != stream.observed_samples[index].class_label
                or prediction.observed_concept_id != trace.concept_ids_by_sample_index[index]
            ):
                raise ValueError("予測の位置/観測/概念が対応していません。")
            expected_transition = transitions[index // chunk] if (index + 1) % chunk == 0 else None
            if prediction.detection_batch_transition != expected_transition:
                raise ValueError("予測に付加された検出結果がbatch記録と対応していません。")
    correctness = tuple(
        tuple(p.prediction_is_correct for p in c.stream_processor.sample_predictions)
        for c in participants.run_clients
    )
    changes = tuple(
        extract_concept_change_sample_indices(
            concept_ids_by_sample_index=t.concept_ids_by_sample_index
        )
        for t in run_result.evaluation_concept_traces
    )
    # FedDriftの切替は検出batch transitionに保持される。共有適応ownerはserver remap用。
    switches = tuple(
        tuple(
            t.last_sample_index
            for t in c.stream_processor.completed_detection_batch_transitions
            if t.adaptation_kind != "current_model_retained"
        )
        for c in participants.run_clients
    )
    if any(index >= processed for indices in switches for index in indices):
        raise ValueError("切替位置は処理した標本の範囲内が必要です。")
    repository = participants.run_server.global_model_repository
    # FedDriftの独立モデルは共有部の重複除外を行わず、各実体を全て数える。
    parameters = tuple(
        parameter
        for mid in repository.global_model_ids
        for parameter in repository.get_global_model_parameters(model_id=mid).values()
    )
    model_diagnostics = derive_feddrift_model_diagnostic_metrics(
        run_clients=participants.run_clients,
        run_server=participants.run_server,
    )
    records = model_diagnostics.synchronized_timestep_model_counts
    if len(records) != run_result.synchronization_interval_count or tuple(
        record.timestep_index for record in records
    ) != tuple(range(run_result.synchronization_interval_count)):
        raise ValueError("モデル数記録の時刻/件数がrunの同期と対応していません。")
    server_counts = participants.run_server.snapshot_server_computation_counts()
    client_consolidation_count = (
        participants.run_server.client_consolidation_parameter_multiply_accumulate_count
    )
    processed_sample_count = processed * len(participants.run_clients)
    summary = (
        summarize_computation_cost(
            model_computation_counts=model_computation_counts,
            server_computation_counts=server_counts,
            processed_sample_count=processed_sample_count,
            held_model_sample_count=held_model_sample_count,
        )
        if model_computation_counts is not None
        else None
    )
    return FeddriftRunMetrics(
        model_diagnostic_metrics=model_diagnostics,
        prediction_accuracy=calculate_prediction_accuracy(
            prediction_correctness_by_client=correctness
        ),
        stable_period_prediction_accuracy=calculate_stable_period_prediction_accuracy(
            prediction_correctness_by_client=correctness,
            concept_change_sample_indices_by_client=changes,
            recovery_window_sample_count=run_metric_settings.post_change_recovery_window_sample_count,
        ),
        training_model_switch_detection_metrics=calculate_detection_metrics(
            concept_change_sample_indices_by_client=changes,
            detection_sample_indices_by_client=switches,
            maximum_delay_sample_count=run_metric_settings.maximum_detection_delay_sample_count,
        ),
        final_global_model_count=len(repository.global_model_ids),
        communication_volume=participants.run_server.communication_volume_record_store.get_state_snapshot(),
        final_parameter_value_count=sum(p.numel() for p in parameters),
        final_parameter_byte_count=sum(p.numel() * p.element_size() for p in parameters),
        model_computation_counts=model_computation_counts,
        server_computation_counts=server_counts,
        processed_sample_count=processed_sample_count,
        held_model_sample_count=held_model_sample_count,
        client_consolidation_parameter_multiply_accumulate_count=client_consolidation_count,
        computation_cost_summary=summary,
    )
