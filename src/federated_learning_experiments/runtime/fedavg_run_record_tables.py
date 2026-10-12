"""FedAvgの予測・同期・計算・学習診断を名前付き配列へ変換する。"""

from dataclasses import fields
from typing import cast

from numpy import asarray, bool_, float64, int64, ndarray, str_

from federated_learning_experiments.core.settings_serialization import (
    convert_settings_to_plain_mapping,
)
from federated_learning_experiments.evaluation.communication_volume_record_store import (
    CommunicationVolumeSnapshot,
)
from federated_learning_experiments.evaluation.run_metric_calculations import (
    extract_concept_change_sample_indices,
)
from federated_learning_experiments.learning.models.model_computation_measurement import (
    ModelComputationCounts,
)
from federated_learning_experiments.methods.fedavg.fedavg_client import FedavgClient
from federated_learning_experiments.runtime.fedavg_experiment_run import (
    FedavgExperimentRun,
    derive_fedavg_run_metrics,
)
from federated_learning_experiments.runtime.fedavg_run_server import FedavgRunServer


def build_fedavg_run_record_tables(*, experiment_run: FedavgExperimentRun) -> dict[str, ndarray]:
    """現在の記録と指標の対応を検査し、保存する表を作る。"""
    if type(experiment_run) is not FedavgExperimentRun:
        raise TypeError("exact FedAvg runが必要です。")
    measured = experiment_run.measured_run
    metrics = derive_fedavg_run_metrics(
        run_settings=experiment_run.run_settings,
        run_result=measured.run_result,
        participants=measured.participants,
        model_computation_counts=measured.model_computation_counts,
    )
    if convert_settings_to_plain_mapping(
        metrics, non_finite_float_as_none=True
    ) != convert_settings_to_plain_mapping(
        experiment_run.run_metrics, non_finite_float_as_none=True
    ):
        raise ValueError("指標が現在の参加者記録と不一致です。")
    clients = cast(tuple[FedavgClient, ...], measured.participants.client_operations)
    server = cast(FedavgRunServer, measured.participants.server_operations)
    result = measured.run_result
    count = len(clients)
    processed = result.processed_sample_count_per_client
    rounds = measured.round_model_computation_counts
    if len(rounds) != result.synchronization_interval_count or tuple(
        r.round_index for r in rounds
    ) != tuple(range(len(rounds))):
        raise ValueError("round計測位置と進行が不一致です。")
    for field in fields(ModelComputationCounts):
        if getattr(measured.model_computation_counts, field.name) != sum(
            getattr(r.local_processing_model_computation_counts, field.name)
            + getattr(r.synchronization_model_computation_counts, field.name)
            for r in rounds
        ) + getattr(measured.finalization_model_computation_counts, field.name):
            raise ValueError("round計測の合計と本計数が不一致です。")
    tables: dict[str, ndarray] = {}
    for column in (
        "sample_index",
        "predicted_class_id",
        "observed_class_id",
        "observed_concept_id",
        "prediction_is_correct",
        "prediction_model_id",
    ):
        tables[f"sample_prediction.{column}"] = asarray(
            [[getattr(p, column) for p in c.sample_predictions] for c in clients],
            dtype=bool_ if column == "prediction_is_correct" else int64,
        ).reshape(count, processed)
    tables["sample_prediction.held_model_count_at_prediction"] = asarray(
        [[1] * processed for _ in clients], dtype=int64
    ).reshape(count, processed)
    tables["generated_concept.concept_id"] = asarray(
        [trace.concept_ids_by_sample_index for trace in result.evaluation_concept_traces],
        dtype=int64,
    )
    changes = [
        (t.client_id, i)
        for t in result.evaluation_concept_traces
        for i in extract_concept_change_sample_indices(
            concept_ids_by_sample_index=t.concept_ids_by_sample_index
        )
    ]
    for index, name in enumerate(("client_id", "sample_index")):
        tables[f"concept_change.{name}"] = asarray([row[index] for row in changes], dtype=int64)
    tables["synchronization.round_index"] = asarray([r.round_index for r in rounds], dtype=int64)
    tables["synchronization.global_model_count"] = asarray([1] * len(rounds), dtype=int64)
    tables["synchronization.client_held_model_count"] = asarray(
        [[1] * count for _ in rounds], dtype=int64
    ).reshape(len(rounds), count)
    for field in fields(CommunicationVolumeSnapshot):
        cumulative = [
            getattr(snapshot, field.name) for snapshot in server.synchronized_communication_volumes
        ]
        tables[f"synchronization.communication_volume.{field.name}"] = asarray(
            [value - (cumulative[i - 1] if i else 0) for i, value in enumerate(cumulative)],
            dtype=int64,
        )
    for phase, attribute in (
        ("local_processing", "local_processing_model_computation_counts"),
        ("synchronization", "synchronization_model_computation_counts"),
    ):
        for field in fields(ModelComputationCounts):
            tables[f"round_model_computation.{phase}.{field.name}"] = asarray(
                [getattr(getattr(r, attribute), field.name) for r in rounds], dtype=int64
            )
    for phase, counts in (
        ("preparation", measured.preparation_model_computation_counts),
        ("finalization", measured.finalization_model_computation_counts),
    ):
        for field in fields(ModelComputationCounts):
            tables[f"model_computation.{phase}.{field.name}"] = asarray(
                [getattr(counts, field.name)], dtype=int64
            )
    aggregation = server.synchronized_aggregation_parameter_counts
    tables["synchronization.aggregation_parameter_multiply_accumulate_count"] = asarray(
        [value - (aggregation[i - 1] if i else 0) for i, value in enumerate(aggregation)],
        dtype=int64,
    )
    tables["model_training.client_id"] = asarray([c.client_id for c in clients], dtype=int64)
    tables["model_training.model_id"] = asarray([0] * count, dtype=int64)
    for name in (
        "trained_sample_count",
        "parameter_update_step_count",
        "prediction_forward_call_count",
        "statistics_forward_call_count",
    ):
        tables[f"model_training.{name}"] = asarray([getattr(c, name) for c in clients], dtype=int64)
    for name in ("observed_loss_count", "mean_loss", "sum_squared_loss_deviations"):
        moments = []
        for client in clients:
            statistics = client.loss_statistics_store.get_model_loss_statistics(model_id=0)
            if statistics is None:
                raise ValueError("client統計がありません。")
            moments.append(getattr(statistics.overall_loss_moments, name))
        tables[f"model_loss_statistics.{name}"] = asarray(
            moments, dtype=int64 if name == "observed_loss_count" else float64
        )
    for name in (
        "generated_sample_count_per_client",
        "processed_sample_count_per_client",
        "unprocessed_tail_sample_count_per_client",
        "synchronization_interval_count",
    ):
        tables[f"run_progress.{name}"] = asarray([getattr(result, name)], dtype=int64)
    for name in ("stage_name", "client_id", "sample_index", "round_index"):
        tables[f"execution_event.{name}"] = asarray(
            [
                getattr(e, name) if getattr(e, name) is not None else -1
                for e in result.execution_events
            ],
            dtype=str_ if name == "stage_name" else int64,
        )
    return tables
