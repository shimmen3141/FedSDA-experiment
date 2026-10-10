"""最終構成のFedSDAの参加者の記録から、モデル別の診断（モデル数、モデル別の学習量、クラスタリングの判定と真の概念の一致）の指標を導出する。

読取りだけを行い、参加者の状態と乱数を変えない。
"""

from dataclasses import dataclass

from numpy import asarray, float64

from federated_learning_experiments.evaluation.model_diagnostic_metric_calculations import (
    ClusterAssignmentTrueConceptAgreement,
    ValueDistributionSummary,
    calculate_cluster_assignment_true_concept_agreement,
    calculate_lower_value_same_concept_auc,
    summarize_value_distribution,
)
from federated_learning_experiments.runtime.fedsda_run_client import FedsdaRunClient
from federated_learning_experiments.runtime.fedsda_run_server import FedsdaRunServer


@dataclass(frozen=True, kw_only=True)
class ServerRoundModelCounts:
    """サーバの同期のラウンドごとの、同期の最後の配布の後のモデル数。"""

    # ラウンドごとの、グローバルモデル（正式IDのモデル）の数。
    global_model_counts: tuple[int, ...]
    # ラウンドごと・clientごとの、保有モデルの数（まだ登録していない一時IDのモデルを含む）。
    client_held_model_counts: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, kw_only=True)
class GlobalModelCountMetrics:
    """サーバの同期のラウンドごとの、同期の後のグローバルモデル数のまとめ。ラウンドがなければ、すべて0。"""

    mean_global_model_count_after_synchronization: float
    maximum_global_model_count_after_synchronization: float
    # ラウンドごとのグローバルモデル数の合計（モデル数×ラウンドの延べ数）。
    total_global_model_count_over_server_rounds: float


@dataclass(frozen=True, kw_only=True)
class ClientHeldModelTrainingAmount:
    """clientが、実行の終わりに保有するモデル1つの、学習量。"""

    client_id: int
    model_id: int
    # そのモデルへ割り当てられ、clientが保持している学習標本の数。
    stored_training_sample_count: int
    # そのモデルの学習に使った標本の延べ数。
    trained_sample_count: int
    # そのモデルの、パラメータ更新の回数。
    parameter_update_step_count: int


@dataclass(frozen=True, kw_only=True)
class HeldModelTrainingAmountMetrics:
    """保有モデル別の学習量（モデルごとに、全clientで足した値）の、モデルの間のまとめ。"""

    stored_training_sample_count_summary: ValueDistributionSummary
    trained_sample_count_summary: ValueDistributionSummary
    parameter_update_step_count_summary: ValueDistributionSummary


@dataclass(frozen=True, kw_only=True)
class ClusteringTrueConceptAgreementMetrics:
    """クラスタリングの判定と、真の概念の一致の比較（真の概念の一致・不一致が分かるモデル対について）。

    真の概念は、診断のためだけに使う（判定には使われていない）。AUCは、小さい距離を、同じ概念の証拠と
    みなしたROC-AUCで、真の概念が一致する対か、一致しない対が、なければ、定義できないので、None。
    """

    true_concept_known_pair_count: int
    same_true_concept_pair_count: int
    cluster_assignment_agreement: ClusterAssignmentTrueConceptAgreement
    loss_increase_distance_same_concept_auc: float | None
    concept_specific_parameter_distance_same_concept_auc: float | None


@dataclass(frozen=True, kw_only=True)
class FedsdaModelDiagnosticMetrics:
    """モデル別の診断の指標。"""

    global_model_count_metrics: GlobalModelCountMetrics
    held_model_training_amount_metrics: HeldModelTrainingAmountMetrics
    clustering_true_concept_agreement_metrics: ClusteringTrueConceptAgreementMetrics


def _validate_run_clients(*, run_clients: object) -> None:
    if type(run_clients) is not tuple:
        raise TypeError("run_clients must be exact tuple")
    for run_client in run_clients:
        if type(run_client) is not FedsdaRunClient:
            raise TypeError("run_clients must hold exact FedsdaRunClient")


def _validate_run_server(*, run_server: object) -> None:
    if type(run_server) is not FedsdaRunServer:
        raise TypeError("run_server must be exact FedsdaRunServer")


def count_models_after_server_round_synchronizations(
    *, run_clients: tuple[FedsdaRunClient, ...], run_server: FedsdaRunServer
) -> ServerRoundModelCounts:
    """サーバの同期のラウンドごとに、同期の最後の配布で各clientが受け取った後の、保有モデルのIDから数える。

    グローバルモデルは、正式ID（0以上）のモデルで、全clientが同じものを受け取る。clientは、そのほかに、
    まだ登録していない一時ID（負）のモデルを持つことがある。
    """
    _validate_run_clients(run_clients=run_clients)
    _validate_run_server(run_server=run_server)
    global_model_counts: list[int] = []
    client_held_model_counts: list[tuple[int, ...]] = []
    for synchronization in run_server.snapshot_server_round_synchronizations():
        held_model_ids_by_client = [
            distribution_application.held_model_ids
            for distribution_application in synchronization.distribution_applications
        ]
        if len(held_model_ids_by_client) != len(run_clients):
            raise ValueError("every client must have a distribution application in each round")
        distributed_global_model_ids_by_client = {
            tuple(model_id for model_id in held_model_ids if model_id >= 0)
            for held_model_ids in held_model_ids_by_client
        }
        if len(distributed_global_model_ids_by_client) != 1:
            raise ValueError("all clients must hold the same global models after a distribution")
        global_model_counts.append(len(next(iter(distributed_global_model_ids_by_client))))
        client_held_model_counts.append(
            tuple(len(held_model_ids) for held_model_ids in held_model_ids_by_client)
        )
    return ServerRoundModelCounts(
        global_model_counts=tuple(global_model_counts),
        client_held_model_counts=tuple(client_held_model_counts),
    )


def collect_client_held_model_training_amounts(
    *, run_clients: tuple[FedsdaRunClient, ...]
) -> tuple[ClientHeldModelTrainingAmount, ...]:
    """clientの順、clientの保有の順で、実行の終わりの保有モデルごとの学習量を返す。

    保有モデルは、まだ登録していない一時IDのモデルを含む。標本や計数の記録がないモデルの値は、0。
    """
    _validate_run_clients(run_clients=run_clients)
    training_amounts: list[ClientHeldModelTrainingAmount] = []
    for run_client in run_clients:
        owners = run_client.owners
        stored_training_sample_counts_by_model_id = {
            training_sample_collection.model_id: len(training_sample_collection.training_samples)
            for training_sample_collection in owners.training_sample_store.snapshot_ordered_model_training_samples()
        }
        counts_snapshot = owners.model_training_and_assignment_counts_store.snapshot_model_training_and_assignment_counts()
        for held_model_training_state in (
            owners.held_model_training_state_registry.snapshot_ordered_held_model_training_states()
        ):
            model_id = held_model_training_state.model_id
            training_amounts.append(
                ClientHeldModelTrainingAmount(
                    client_id=run_client.client_id,
                    model_id=model_id,
                    stored_training_sample_count=stored_training_sample_counts_by_model_id.get(
                        model_id, 0
                    ),
                    trained_sample_count=counts_snapshot.trained_sample_counts_by_model_id.get(
                        model_id, 0
                    ),
                    parameter_update_step_count=(
                        counts_snapshot.parameter_update_step_counts_by_model_id.get(model_id, 0)
                    ),
                )
            )
    return tuple(training_amounts)


def _summarize_training_amount_by_model(
    *, training_amounts: tuple[ClientHeldModelTrainingAmount, ...], amount_name: str
) -> ValueDistributionSummary:
    """モデルごとに、全clientの値を足し（モデルは、最初に現れた順）、モデルの間でまとめる。"""
    amounts_by_model_id: dict[int, int] = {}
    for training_amount in training_amounts:
        amounts_by_model_id[training_amount.model_id] = amounts_by_model_id.get(
            training_amount.model_id, 0
        ) + getattr(training_amount, amount_name)
    return summarize_value_distribution(values=tuple(amounts_by_model_id.values()))


def derive_fedsda_model_diagnostic_metrics(
    *, run_clients: tuple[FedsdaRunClient, ...], run_server: FedsdaRunServer
) -> FedsdaModelDiagnosticMetrics:
    """参加者の記録から、モデル別の診断の指標を作る。"""
    _validate_run_clients(run_clients=run_clients)
    _validate_run_server(run_server=run_server)

    # ---- サーバの同期のラウンドごとの、グローバルモデル数のまとめ ----
    global_model_counts = count_models_after_server_round_synchronizations(
        run_clients=run_clients, run_server=run_server
    ).global_model_counts
    if global_model_counts:
        global_model_count_array = asarray(global_model_counts, dtype=float64)
        global_model_count_metrics = GlobalModelCountMetrics(
            mean_global_model_count_after_synchronization=float(global_model_count_array.mean()),
            maximum_global_model_count_after_synchronization=float(global_model_count_array.max()),
            total_global_model_count_over_server_rounds=float(global_model_count_array.sum()),
        )
    else:
        global_model_count_metrics = GlobalModelCountMetrics(
            mean_global_model_count_after_synchronization=0.0,
            maximum_global_model_count_after_synchronization=0.0,
            total_global_model_count_over_server_rounds=0.0,
        )

    # ---- 保有モデル別の学習量 ----
    training_amounts = collect_client_held_model_training_amounts(run_clients=run_clients)
    held_model_training_amount_metrics = HeldModelTrainingAmountMetrics(
        stored_training_sample_count_summary=_summarize_training_amount_by_model(
            training_amounts=training_amounts, amount_name="stored_training_sample_count"
        ),
        trained_sample_count_summary=_summarize_training_amount_by_model(
            training_amounts=training_amounts, amount_name="trained_sample_count"
        ),
        parameter_update_step_count_summary=_summarize_training_amount_by_model(
            training_amounts=training_amounts, amount_name="parameter_update_step_count"
        ),
    )

    # ---- クラスタリングの判定と、真の概念の一致の比較（真の概念の一致・不一致が分かる対だけ） ----
    known_pair_observations = [
        (pair_observation, pair_observation.true_concepts_match)
        for pair_observation in run_server.owners.model_clustering_record_store.snapshot_pair_clustering_observations()
        if pair_observation.true_concepts_match is not None
    ]
    cluster_assignment_agreement = calculate_cluster_assignment_true_concept_agreement(
        assigned_to_same_cluster_flags=tuple(
            pair_observation.assigned_to_same_cluster
            for pair_observation, _ in known_pair_observations
        ),
        true_concepts_match_flags=tuple(
            true_concepts_match for _, true_concepts_match in known_pair_observations
        ),
    )

    def calculate_distance_auc(distance_name: str) -> float | None:
        return calculate_lower_value_same_concept_auc(
            same_concept_values=tuple(
                getattr(pair_observation, distance_name)
                for pair_observation, true_concepts_match in known_pair_observations
                if true_concepts_match
            ),
            different_concept_values=tuple(
                getattr(pair_observation, distance_name)
                for pair_observation, true_concepts_match in known_pair_observations
                if not true_concepts_match
            ),
        )

    clustering_true_concept_agreement_metrics = ClusteringTrueConceptAgreementMetrics(
        true_concept_known_pair_count=len(known_pair_observations),
        same_true_concept_pair_count=(
            cluster_assignment_agreement.same_cluster_same_concept_pair_count
            + cluster_assignment_agreement.different_cluster_same_concept_pair_count
        ),
        cluster_assignment_agreement=cluster_assignment_agreement,
        loss_increase_distance_same_concept_auc=calculate_distance_auc("loss_increase_distance"),
        concept_specific_parameter_distance_same_concept_auc=calculate_distance_auc(
            "concept_specific_parameter_distance"
        ),
    )
    return FedsdaModelDiagnosticMetrics(
        global_model_count_metrics=global_model_count_metrics,
        held_model_training_amount_metrics=held_model_training_amount_metrics,
        clustering_true_concept_agreement_metrics=clustering_true_concept_agreement_metrics,
    )
