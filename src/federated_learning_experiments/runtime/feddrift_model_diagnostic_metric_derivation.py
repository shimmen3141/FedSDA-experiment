"""FedDriftの参加者の記録からモデル診断を読取りで導出する。"""

from dataclasses import dataclass

from numpy import asarray, float64

from federated_learning_experiments.evaluation.feddrift_model_diagnostic_records import (
    FeddriftClientHeldModelTrainingAmount,
    FeddriftSynchronizedTimestepModelCounts,
)
from federated_learning_experiments.evaluation.model_clustering_record_store import (
    ModelClusteringRecordStore,
    ModelPairClusteringObservation,
)
from federated_learning_experiments.evaluation.model_diagnostic_metric_calculations import (
    ClusterAssignmentTrueConceptAgreement,
    ValueDistributionSummary,
    calculate_cluster_assignment_true_concept_agreement,
    calculate_lower_value_same_concept_auc,
    summarize_value_distribution,
)
from federated_learning_experiments.learning.training.held_model_training_state_registry import (
    HeldModelTrainingState,
    HeldModelTrainingStateRegistry,
)
from federated_learning_experiments.learning.training.model_training_and_assignment_counts import (
    ModelTrainingAndAssignmentCountsStore,
)
from federated_learning_experiments.learning.training.model_training_sample_store import (
    ModelTrainingSampleStore,
)
from federated_learning_experiments.runtime.feddrift_run_client import FeddriftRunClient
from federated_learning_experiments.runtime.feddrift_run_server import FeddriftRunServer


@dataclass(frozen=True, kw_only=True)
class FeddriftModelDiagnosticMetrics:
    """モデル数の系列、終了時学習量とクラスタの真概念診断。"""

    synchronized_timestep_model_counts: tuple[FeddriftSynchronizedTimestepModelCounts, ...]
    client_held_model_training_amounts: tuple[FeddriftClientHeldModelTrainingAmount, ...]
    mean_global_model_count_after_synchronization: float
    maximum_global_model_count_after_synchronization: float
    total_global_model_count_over_synchronized_timesteps: float
    stored_training_sample_count_summary: ValueDistributionSummary
    trained_sample_count_summary: ValueDistributionSummary
    parameter_update_step_count_summary: ValueDistributionSummary
    true_concept_known_pair_count: int
    same_true_concept_pair_count: int
    cluster_assignment_agreement: ClusterAssignmentTrueConceptAgreement
    loss_increase_distance_same_concept_auc: float | None
    concept_specific_parameter_distance_same_concept_auc: float | None


def derive_feddrift_model_diagnostic_metrics(
    *,
    run_clients: tuple[FeddriftRunClient, ...],
    run_server: FeddriftRunServer,
) -> FeddriftModelDiagnosticMetrics:
    """参加者の同一性と記録の契約を検査して診断を導出する。"""
    if (
        type(run_server) is not FeddriftRunServer
        or type(run_clients) is not tuple
        or not run_clients
    ):
        raise TypeError("exact serverと非空client tupleが必要です。")
    if len(run_clients) != len(run_server.run_clients) or any(
        type(client) is not FeddriftRunClient or client is not registered
        for client, registered in zip(run_clients, run_server.run_clients, strict=True)
    ):
        raise ValueError("serverと同じ実clientを同じ順に指定してください。")
    ids = [client.client_id for client in run_clients]
    if any(type(identifier) is not int or identifier < 0 for identifier in ids) or len(
        set(ids)
    ) != len(ids):
        raise ValueError("client IDは非負builtin intで一意が必要です。")
    synchronized_counts = run_server.snapshot_synchronized_timestep_model_counts()
    amounts: list[FeddriftClientHeldModelTrainingAmount] = []
    for client in run_clients:
        owners = client.owners
        for owner, expected in (
            (owners.held_model_training_state_registry, HeldModelTrainingStateRegistry),
            (owners.training_sample_store, ModelTrainingSampleStore),
            (
                owners.model_training_and_assignment_counts_store,
                ModelTrainingAndAssignmentCountsStore,
            ),
        ):
            if type(owner) is not expected:
                raise TypeError(f"exact {expected.__name__}が必要です。")
        stored_counts = {}
        for collection in owners.training_sample_store.snapshot_ordered_model_training_samples():
            if type(collection.model_id) is not int:
                raise TypeError("model IDはbuiltin intが必要です。")
            stored_counts[collection.model_id] = len(collection.training_samples)
        counts = owners.model_training_and_assignment_counts_store.snapshot_model_training_and_assignment_counts()
        for mapping in (
            counts.trained_sample_counts_by_model_id,
            counts.parameter_update_step_counts_by_model_id,
        ):
            for identifier, count in mapping.items():
                if type(identifier) is not int or type(count) is not int:
                    raise TypeError("model IDと学習計数はbuiltin intが必要です。")
                if count < 0:
                    raise ValueError("学習計数は非負が必要です。")
        for state in (
            owners.held_model_training_state_registry.snapshot_ordered_held_model_training_states()
        ):
            if type(state) is not HeldModelTrainingState:
                raise TypeError("exact HeldModelTrainingStateが必要です。")
            amounts.append(
                FeddriftClientHeldModelTrainingAmount(
                    client_id=client.client_id,
                    model_id=state.model_id,
                    stored_training_sample_count=stored_counts.get(state.model_id, 0),
                    trained_sample_count=counts.trained_sample_counts_by_model_id.get(
                        state.model_id, 0
                    ),
                    parameter_update_step_count=counts.parameter_update_step_counts_by_model_id.get(
                        state.model_id, 0
                    ),
                )
            )

    def summarize_amount(name: str) -> ValueDistributionSummary:
        totals: dict[int, int] = {}
        for amount in amounts:
            totals[amount.model_id] = totals.get(amount.model_id, 0) + getattr(amount, name)
        return summarize_value_distribution(values=tuple(totals.values()))

    store = run_server.model_clustering_record_store
    if type(store) is not ModelClusteringRecordStore:
        raise TypeError("exact ModelClusteringRecordStoreが必要です。")
    known: list[ModelPairClusteringObservation] = []
    for pair in store.snapshot_pair_clustering_observations():
        if type(pair) is not ModelPairClusteringObservation:
            raise TypeError("exact ModelPairClusteringObservationが必要です。")
        pair.__post_init__()
        if pair.true_concepts_match is not None:
            known.append(pair)
    agreement = calculate_cluster_assignment_true_concept_agreement(
        assigned_to_same_cluster_flags=tuple(pair.assigned_to_same_cluster for pair in known),
        true_concepts_match_flags=tuple(pair.true_concepts_match is True for pair in known),
    )

    def distance_auc(name: str) -> float | None:
        return calculate_lower_value_same_concept_auc(
            same_concept_values=tuple(
                getattr(pair, name) for pair in known if pair.true_concepts_match
            ),
            different_concept_values=tuple(
                getattr(pair, name) for pair in known if not pair.true_concepts_match
            ),
        )

    globals_array = asarray(
        [record.global_model_count for record in synchronized_counts], dtype=float64
    )
    return FeddriftModelDiagnosticMetrics(
        synchronized_timestep_model_counts=synchronized_counts,
        client_held_model_training_amounts=tuple(amounts),
        mean_global_model_count_after_synchronization=float(globals_array.mean())
        if globals_array.size
        else 0.0,
        maximum_global_model_count_after_synchronization=float(globals_array.max())
        if globals_array.size
        else 0.0,
        total_global_model_count_over_synchronized_timesteps=float(globals_array.sum()),
        stored_training_sample_count_summary=summarize_amount("stored_training_sample_count"),
        trained_sample_count_summary=summarize_amount("trained_sample_count"),
        parameter_update_step_count_summary=summarize_amount("parameter_update_step_count"),
        true_concept_known_pair_count=len(known),
        same_true_concept_pair_count=sum(pair.true_concepts_match is True for pair in known),
        cluster_assignment_agreement=agreement,
        loss_increase_distance_same_concept_auc=distance_auc("loss_increase_distance"),
        concept_specific_parameter_distance_same_concept_auc=distance_auc(
            "concept_specific_parameter_distance"
        ),
    )
