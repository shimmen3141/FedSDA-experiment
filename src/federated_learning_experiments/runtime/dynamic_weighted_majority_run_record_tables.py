"""DWMの予測・同期・計算・学習・専門家の寿命を名前付き配列へ変換する。

表の名前と形はFedAvgの記録の表に合わせる（design「Data Models」の対応表）。DWMに無い`model_loss_statistics`は作らず、
`expert_lifetime`を足す。`model_training`は専門家ごと（消された専門家を含む）に1行。
"""

from dataclasses import fields
from typing import cast

from numpy import asarray, bool_, int64, ndarray, str_

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
from federated_learning_experiments.methods.dynamic_weighted_majority.dynamic_weighted_majority_client import (
    DynamicWeightedMajorityClient,
)
from federated_learning_experiments.runtime.dynamic_weighted_majority_experiment_run import (
    DynamicWeightedMajorityExperimentRun,
    derive_dynamic_weighted_majority_run_metrics,
)
from federated_learning_experiments.runtime.dynamic_weighted_majority_run_server import (
    DynamicWeightedMajorityRunServer,
)


def build_dynamic_weighted_majority_run_record_tables(
    *, experiment_run: DynamicWeightedMajorityExperimentRun
) -> dict[str, ndarray]:
    """現在の記録と指標の対応を検査し、保存する表を作る。"""
    if type(experiment_run) is not DynamicWeightedMajorityExperimentRun:
        raise TypeError("exact DWM runが必要です。")
    measured = experiment_run.measured_run
    metrics = derive_dynamic_weighted_majority_run_metrics(
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
    clients = cast(
        tuple[DynamicWeightedMajorityClient, ...], measured.participants.client_operations
    )
    server = cast(DynamicWeightedMajorityRunServer, measured.participants.server_operations)
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
        [[p.held_model_count_at_prediction for p in c.sample_predictions] for c in clients],
        dtype=int64,
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
    if len(server.synchronized_client_expert_counts) != len(rounds):
        raise ValueError("同期の専門家の数の記録とround数が不一致です。")
    tables["synchronization.global_model_count"] = asarray(
        [sum(counts) for counts in server.synchronized_client_expert_counts], dtype=int64
    )
    tables["synchronization.client_held_model_count"] = asarray(
        [list(counts) for counts in server.synchronized_client_expert_counts], dtype=int64
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
    training_rows = [
        (client.client_id, counts) for client in clients for counts in client.expert_training_counts
    ]
    tables["model_training.client_id"] = asarray([row[0] for row in training_rows], dtype=int64)
    tables["model_training.model_id"] = asarray(
        [row[1].expert_id for row in training_rows], dtype=int64
    )
    for name in (
        "trained_sample_count",
        "parameter_update_step_count",
        "prediction_forward_call_count",
    ):
        tables[f"model_training.{name}"] = asarray(
            [getattr(row[1], name) for row in training_rows], dtype=int64
        )
    # DWMは損失の統計のforwardを持たない（列はFedAvgと同じ名前で0）。
    tables["model_training.statistics_forward_call_count"] = asarray(
        [0] * len(training_rows), dtype=int64
    )
    lifetime_rows = [
        (client.client_id, lifetime) for client in clients for lifetime in client.expert_lifetimes
    ]
    tables["expert_lifetime.client_id"] = asarray([row[0] for row in lifetime_rows], dtype=int64)
    for name in ("expert_id", "created_sample_index", "removed_sample_index"):
        tables[f"expert_lifetime.{name}"] = asarray(
            [getattr(row[1], name) for row in lifetime_rows], dtype=int64
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
