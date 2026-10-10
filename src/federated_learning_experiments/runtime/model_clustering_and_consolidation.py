"""クロス評価の結果から、グローバルモデルをクラスタリングし、同じクラスタのモデルを1つへ統合する。"""

from dataclasses import dataclass
from math import inf, sqrt

from torch import Tensor

from federated_learning_experiments.evaluation.model_clustering_record_store import (
    ModelClusteringObservation,
    ModelClusteringRecordStore,
    ModelPairClusteringObservation,
)
from federated_learning_experiments.learning.loss_statistics.bounded_loss_moments import (
    BoundedLossMoments,
)
from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatistics,
)
from federated_learning_experiments.methods.fedsda.consolidation.model_clustering_calculations import (
    ModelClusteringCriteria,
    cluster_model_ids_by_average_linkage,
    compute_classwise_unique_correctness_decision_score,
)
from federated_learning_experiments.methods.fedsda.model_registration.global_model_repository import (
    GlobalModelRepository,
)
from federated_learning_experiments.runtime.fedsda_run_client import FedsdaRunClient
from federated_learning_experiments.runtime.model_cross_evaluation import (
    CrossEvaluationLossSums,
    ModelCrossEvaluation,
)
from federated_learning_experiments.runtime.server_model_registration_and_aggregation import (
    split_shared_and_concept_specific_parameters,
)


@dataclass(frozen=True, kw_only=True)
class ModelConsolidation:
    """クラスタリングと統合1回の結果。"""

    # クラスタ（モデルIDの昇順）の列。先頭のIDの昇順。
    model_clusters: tuple[tuple[int, ...], ...]
    # 統合しなかったら空。統合したら、クラスタリングした全モデルの、ID→代表（クラスタの最小のID）。
    # 1モデルのクラスタの、自分自身への対応を含む。
    model_id_mapping: dict[int, int]
    # 代表へ吸収されて、グローバルモデルから外されたモデルID（昇順）。
    absorbed_model_ids: tuple[int, ...]


def _validate_clustering_and_consolidation_inputs(
    *,
    run_clients: tuple[FedsdaRunClient, ...],
    model_cross_evaluation: ModelCrossEvaluation,
    aggregated_training_sample_counts_by_model_id: dict[int, int],
    global_model_repository: GlobalModelRepository,
    model_clustering_record_store: ModelClusteringRecordStore,
    model_clustering_criteria: ModelClusteringCriteria,
    round_index: int,
) -> None:
    if type(run_clients) is not tuple:
        raise TypeError("run_clients must be builtin tuple")
    for run_client in run_clients:
        if type(run_client) is not FedsdaRunClient:
            raise TypeError("run_clients must hold exact FedsdaRunClient")
    for owner_name, owner, required_type in (
        ("model_cross_evaluation", model_cross_evaluation, ModelCrossEvaluation),
        ("global_model_repository", global_model_repository, GlobalModelRepository),
        (
            "model_clustering_record_store",
            model_clustering_record_store,
            ModelClusteringRecordStore,
        ),
        ("model_clustering_criteria", model_clustering_criteria, ModelClusteringCriteria),
    ):
        if type(owner) is not required_type:
            raise TypeError(f"{owner_name} must be exact {required_type.__name__}")
    # frozenを回避して組み立てた値も拒否できるよう、基準の検査をもう一度行う。
    model_clustering_criteria.__post_init__()
    if type(round_index) is not int:
        raise TypeError("round_index must be builtin int")
    if round_index < 0:
        raise ValueError("round_index must be nonnegative")
    clustered_model_ids = model_cross_evaluation.cross_evaluated_model_ids
    # モデルが1つ以下なら、クラスタリングの対象がない（呼出し側が、クロス評価もクラスタリングも行わない）。
    if type(clustered_model_ids) is not tuple or len(clustered_model_ids) < 2:
        raise ValueError("model_cross_evaluation must hold a tuple of two or more model IDs")
    held_global_model_ids = global_model_repository.global_model_ids
    for model_id in clustered_model_ids:
        if type(model_id) is not int or model_id not in held_global_model_ids:
            raise ValueError(
                "model_cross_evaluation model IDs must be global model IDs with parameters"
            )
    if len(set(clustered_model_ids)) != len(clustered_model_ids):
        raise ValueError("model_cross_evaluation model IDs must not hold duplicates")
    loss_sums_table = model_cross_evaluation.loss_sums_by_candidate_and_target_model_id
    if type(loss_sums_table) is not dict:
        raise TypeError("model_cross_evaluation loss sums must be builtin dict")
    for candidate_model_id in clustered_model_ids:
        for target_model_id in clustered_model_ids:
            if (
                type(loss_sums_table.get((candidate_model_id, target_model_id)))
                is not CrossEvaluationLossSums
            ):
                raise ValueError(
                    "model_cross_evaluation must hold loss sums of every pair of its model IDs"
                )
    if type(model_cross_evaluation.unique_correctness_counts_by_model_pair) is not dict:
        raise TypeError("model_cross_evaluation unique correctness counts must be builtin dict")
    if type(aggregated_training_sample_counts_by_model_id) is not dict:
        raise TypeError("aggregated_training_sample_counts_by_model_id must be builtin dict")
    for model_id in clustered_model_ids:
        if type(aggregated_training_sample_counts_by_model_id.get(model_id)) is not int:
            raise ValueError(
                "aggregated_training_sample_counts_by_model_id must hold a builtin int count "
                "of every clustered model"
            )


def _get_unique_majority_concept_ids(
    *, run_clients: tuple[FedsdaRunClient, ...], model_ids: tuple[int, ...]
) -> dict[int, int]:
    """モデルごとの、全clientの割当概念の計数を足したときの、最多の概念（一意なときだけ）。診断用。"""
    majority_concept_id_by_model_id: dict[int, int] = {}
    for model_id in model_ids:
        assigned_sample_counts_by_concept_id: dict[int, int] = {}
        for run_client in run_clients:
            for (
                concept_id,
                assigned_sample_count,
            ) in run_client.get_model_assigned_sample_concept_counts(model_id=model_id).items():
                assigned_sample_counts_by_concept_id[concept_id] = (
                    assigned_sample_counts_by_concept_id.get(concept_id, 0) + assigned_sample_count
                )
        if not assigned_sample_counts_by_concept_id:
            continue
        maximum_assigned_sample_count = max(assigned_sample_counts_by_concept_id.values())
        majority_concept_ids = [
            concept_id
            for concept_id, assigned_sample_count in assigned_sample_counts_by_concept_id.items()
            if assigned_sample_count == maximum_assigned_sample_count
        ]
        if len(majority_concept_ids) == 1:
            majority_concept_id_by_model_id[model_id] = majority_concept_ids[0]
    return majority_concept_id_by_model_id


def _compute_concept_specific_parameter_distance(
    *, left_parameters: dict[str, Tensor], right_parameters: dict[str, Tensor]
) -> float:
    """概念固有部のパラメータの、相対L2距離（倍精度）。差の2乗和÷（2乗和の和の半分）の平方根。診断用。"""
    difference_square_sum = 0.0
    scale_square_sum = 0.0
    for parameter_name, left_parameter_values in left_parameters.items():
        left_values = left_parameter_values.detach().double()
        right_values = right_parameters[parameter_name].detach().double()
        difference = left_values - right_values
        difference_square_sum += float((difference * difference).sum().item())
        scale_square_sum += float((left_values**2).sum().item() + (right_values**2).sum().item())
    if scale_square_sum == 0.0:
        return 0.0 if difference_square_sum == 0.0 else inf
    return sqrt(difference_square_sum / (0.5 * scale_square_sum))


def _compute_mean_bounded_loss(loss_sums: CrossEvaluationLossSums) -> float:
    return loss_sums.bounded_loss_sum / loss_sums.evaluated_sample_count


def _make_model_clustering_observations(
    *,
    round_index: int,
    clustered_model_ids: tuple[int, ...],
    loss_increase_distances_by_model_pair: dict[tuple[int, int], float],
    model_cluster_by_model_id: dict[int, tuple[int, ...]],
) -> tuple[ModelClusteringObservation, ...]:
    model_clustering_observations = []
    for model_id in clustered_model_ids:
        distances_to_other_models = [
            (loss_increase_distance, other_model_id)
            for model_pair, loss_increase_distance in loss_increase_distances_by_model_pair.items()
            if model_id in model_pair
            for other_model_id in model_pair
            if other_model_id != model_id
        ]
        nearest_loss_increase_distance, nearest_model_id = (
            min(distances_to_other_models) if distances_to_other_models else (None, None)
        )
        model_cluster = model_cluster_by_model_id[model_id]
        representative_model_id = min(model_cluster)
        within_cluster_distances = [
            loss_increase_distance
            for model_pair, loss_increase_distance in loss_increase_distances_by_model_pair.items()
            if model_pair[0] in model_cluster and model_pair[1] in model_cluster
        ]
        model_clustering_observations.append(
            ModelClusteringObservation(
                round_index=round_index,
                model_id=model_id,
                nearest_model_id=nearest_model_id,
                nearest_loss_increase_distance=nearest_loss_increase_distance,
                representative_model_id=representative_model_id,
                cluster_model_count=len(model_cluster),
                maximum_within_cluster_distance=(
                    max(within_cluster_distances) if within_cluster_distances else None
                ),
                evaluated_within_cluster_pair_count=len(within_cluster_distances),
                possible_within_cluster_pair_count=len(model_cluster)
                * (len(model_cluster) - 1)
                // 2,
                merged_with_other_models=len(model_cluster) > 1,
                absorbed_into_representative=model_id != representative_model_id,
            )
        )
    return tuple(model_clustering_observations)


def _compute_sample_weighted_mean_parameters(
    *,
    model_cluster: tuple[int, ...],
    parameter_snapshots_by_model_id: dict[int, dict[str, Tensor]],
    aggregated_training_sample_counts_by_model_id: dict[int, int],
) -> dict[str, Tensor]:
    """クラスタのメンバーのパラメータの、集約の件数での加重平均。件数がすべて0なら、代表のパラメータ。"""
    sample_weights = {
        model_id: max(aggregated_training_sample_counts_by_model_id[model_id], 0)
        for model_id in model_cluster
    }
    total_sample_weight = sum(sample_weights.values())
    if total_sample_weight <= 0:
        return parameter_snapshots_by_model_id[min(model_cluster)]
    weighted_parameter_sum: dict[str, Tensor] | None = None
    for model_id in model_cluster:
        sample_weight = sample_weights[model_id]
        if sample_weight == 0:
            continue
        member_parameters = parameter_snapshots_by_model_id[model_id]
        if weighted_parameter_sum is None:
            weighted_parameter_sum = {
                parameter_name: parameter_values * sample_weight
                for parameter_name, parameter_values in member_parameters.items()
            }
        else:
            for parameter_name in weighted_parameter_sum:
                weighted_parameter_sum[parameter_name] = (
                    weighted_parameter_sum[parameter_name]
                    + member_parameters[parameter_name] * sample_weight
                )
    if weighted_parameter_sum is None:
        raise ValueError("at least one cluster member must have a positive sample weight")
    return {
        parameter_name: parameter_sum / total_sample_weight
        for parameter_name, parameter_sum in weighted_parameter_sum.items()
    }


def _compute_count_weighted_mean_loss_statistics(
    *, model_cluster: tuple[int, ...], global_model_repository: GlobalModelRepository
) -> ModelAndClassLossStatistics | None:
    """統計を持つメンバーの、件数の合計と、件数での加重平均の平均損失。件数の合計が0なら、置き換えない（None）。"""
    member_loss_moments = [
        member_loss_statistics.overall_loss_moments
        for member_loss_statistics in (
            global_model_repository.get_global_model_loss_statistics(model_id=model_id)
            for model_id in model_cluster
        )
        if member_loss_statistics is not None
    ]
    total_observed_loss_count = sum(
        loss_moments.observed_loss_count for loss_moments in member_loss_moments
    )
    if total_observed_loss_count <= 0:
        return None
    return ModelAndClassLossStatistics(
        overall_loss_moments=BoundedLossMoments(
            observed_loss_count=total_observed_loss_count,
            mean_loss=(
                sum(
                    loss_moments.mean_loss * loss_moments.observed_loss_count
                    for loss_moments in member_loss_moments
                )
                / total_observed_loss_count
            ),
            sum_squared_loss_deviations=0.0,
        )
    )


def cluster_and_consolidate_global_models(
    *,
    run_clients: tuple[FedsdaRunClient, ...],
    model_cross_evaluation: ModelCrossEvaluation,
    aggregated_training_sample_counts_by_model_id: dict[int, int],
    global_model_repository: GlobalModelRepository,
    model_clustering_record_store: ModelClusteringRecordStore,
    model_clustering_criteria: ModelClusteringCriteria,
    round_index: int,
) -> ModelConsolidation:
    """クロス評価したモデル（2つ以上）を、対ごとの判定の値でクラスタリングし、診断の観測を足し、クラスタが減るなら統合する。

    対は、4つの損失の統計（自分どうし2つと、互いの2つ）の件数がすべて下限以上のときだけ、損失の距離を持つ。
    その対に、正誤の集計があれば、判定の値を持つ。判定の値で、average linkageのクラスタを求める。
    統合は、クラスタの最小のID（代表）へ、パラメータ（集約の件数での加重平均）と損失統計（件数での加重平均）を置き、
    ほかのメンバーをグローバルモデルから外す。全部の計算の後で、記録とグローバルモデルを更新する。
    """
    _validate_clustering_and_consolidation_inputs(
        run_clients=run_clients,
        model_cross_evaluation=model_cross_evaluation,
        aggregated_training_sample_counts_by_model_id=aggregated_training_sample_counts_by_model_id,
        global_model_repository=global_model_repository,
        model_clustering_record_store=model_clustering_record_store,
        model_clustering_criteria=model_clustering_criteria,
        round_index=round_index,
    )
    clustered_model_ids = model_cross_evaluation.cross_evaluated_model_ids
    loss_sums_table = model_cross_evaluation.loss_sums_by_candidate_and_target_model_id
    minimum_sample_count = model_clustering_criteria.minimum_pair_evaluation_sample_count
    parameter_snapshots_by_model_id = {
        model_id: global_model_repository.get_global_model_parameters(model_id=model_id)
        for model_id in clustered_model_ids
    }
    concept_specific_parameters_by_model_id = {
        model_id: split_shared_and_concept_specific_parameters(
            parameter_snapshot=parameter_snapshot
        )[1]
        for model_id, parameter_snapshot in parameter_snapshots_by_model_id.items()
    }
    # 真の概念は、診断にだけ使う。判定には渡さない。
    majority_concept_id_by_model_id = _get_unique_majority_concept_ids(
        run_clients=run_clients, model_ids=clustered_model_ids
    )

    loss_increase_distances_by_model_pair: dict[tuple[int, int], float] = {}
    decision_scores_by_model_pair: dict[tuple[int, int], float] = {}
    true_concepts_match_by_model_pair: dict[tuple[int, int], bool] = {}
    parameter_distances_by_model_pair: dict[tuple[int, int], float] = {}
    for first_position, first_model_id in enumerate(clustered_model_ids):
        for second_model_id in clustered_model_ids[first_position + 1 :]:
            lower_model_id = min(first_model_id, second_model_id)
            higher_model_id = max(first_model_id, second_model_id)
            model_pair = (lower_model_id, higher_model_id)
            if (
                lower_model_id in majority_concept_id_by_model_id
                and higher_model_id in majority_concept_id_by_model_id
            ):
                true_concepts_match_by_model_pair[model_pair] = (
                    majority_concept_id_by_model_id[lower_model_id]
                    == majority_concept_id_by_model_id[higher_model_id]
                )
            parameter_distances_by_model_pair[model_pair] = (
                _compute_concept_specific_parameter_distance(
                    left_parameters=concept_specific_parameters_by_model_id[lower_model_id],
                    right_parameters=concept_specific_parameters_by_model_id[higher_model_id],
                )
            )
            lower_on_lower = loss_sums_table[(lower_model_id, lower_model_id)]
            lower_on_higher = loss_sums_table[(lower_model_id, higher_model_id)]
            higher_on_higher = loss_sums_table[(higher_model_id, higher_model_id)]
            higher_on_lower = loss_sums_table[(higher_model_id, lower_model_id)]
            if any(
                loss_sums.evaluated_sample_count < minimum_sample_count
                for loss_sums in (
                    lower_on_lower,
                    lower_on_higher,
                    higher_on_higher,
                    higher_on_lower,
                )
            ):
                continue
            # 互いの標本での平均損失が、自分の標本での平均損失より、どれだけ増えるか。大きいほう。
            loss_increase_distances_by_model_pair[model_pair] = max(
                _compute_mean_bounded_loss(lower_on_higher)
                - _compute_mean_bounded_loss(lower_on_lower),
                _compute_mean_bounded_loss(higher_on_lower)
                - _compute_mean_bounded_loss(higher_on_higher),
            )
            unique_correctness_counts = (
                model_cross_evaluation.unique_correctness_counts_by_model_pair.get(model_pair)
            )
            if unique_correctness_counts is None:
                continue
            decision_scores_by_model_pair[model_pair] = (
                compute_classwise_unique_correctness_decision_score(
                    overall_unique_correctness_counts=(
                        unique_correctness_counts.evaluated_sample_count,
                        unique_correctness_counts.lower_id_model_only_correct_count,
                        unique_correctness_counts.higher_id_model_only_correct_count,
                    ),
                    class_unique_correctness_counts=unique_correctness_counts.class_counts,
                    minimum_class_sample_count=minimum_sample_count,
                    confidence_level=model_clustering_criteria.clustering_confidence_level,
                )
            )
    model_clusters = cluster_model_ids_by_average_linkage(
        model_ids=clustered_model_ids,
        decision_scores_by_model_pair=decision_scores_by_model_pair,
        maximum_same_cluster_decision_score=(
            model_clustering_criteria.maximum_same_cluster_decision_score
        ),
    )
    model_cluster_by_model_id = {
        model_id: model_cluster for model_cluster in model_clusters for model_id in model_cluster
    }

    # 診断の観測（距離を持つ対だけ。対のIDの昇順）。
    pair_clustering_observations = tuple(
        ModelPairClusteringObservation(
            round_index=round_index,
            lower_model_id=model_pair[0],
            higher_model_id=model_pair[1],
            loss_increase_distance=loss_increase_distance,
            decision_score=decision_scores_by_model_pair.get(model_pair, loss_increase_distance),
            assigned_to_same_cluster=(
                model_cluster_by_model_id[model_pair[0]] == model_cluster_by_model_id[model_pair[1]]
            ),
            true_concepts_match=true_concepts_match_by_model_pair.get(model_pair),
            concept_specific_parameter_distance=parameter_distances_by_model_pair[model_pair],
        )
        for model_pair, loss_increase_distance in sorted(
            loss_increase_distances_by_model_pair.items()
        )
    )
    model_clustering_observations = _make_model_clustering_observations(
        round_index=round_index,
        clustered_model_ids=clustered_model_ids,
        loss_increase_distances_by_model_pair=loss_increase_distances_by_model_pair,
        model_cluster_by_model_id=model_cluster_by_model_id,
    )

    # 統合の計算（クラスタが減るときだけ）。グローバルモデルは、まだ変えない。
    consolidates_models = len(model_clusters) < len(clustered_model_ids)
    model_id_mapping: dict[int, int] = {}
    consolidated_parameters_by_representative_model_id: dict[int, dict[str, Tensor]] = {}
    consolidated_loss_statistics_by_representative_model_id: dict[
        int, ModelAndClassLossStatistics
    ] = {}
    if consolidates_models:
        for model_cluster in model_clusters:
            representative_model_id = min(model_cluster)
            for model_id in model_cluster:
                model_id_mapping[model_id] = representative_model_id
            if len(model_cluster) <= 1:
                continue
            consolidated_parameters_by_representative_model_id[representative_model_id] = (
                _compute_sample_weighted_mean_parameters(
                    model_cluster=model_cluster,
                    parameter_snapshots_by_model_id=parameter_snapshots_by_model_id,
                    aggregated_training_sample_counts_by_model_id=aggregated_training_sample_counts_by_model_id,
                )
            )
            consolidated_loss_statistics = _compute_count_weighted_mean_loss_statistics(
                model_cluster=model_cluster, global_model_repository=global_model_repository
            )
            if consolidated_loss_statistics is not None:
                consolidated_loss_statistics_by_representative_model_id[representative_model_id] = (
                    consolidated_loss_statistics
                )
    absorbed_model_ids = tuple(
        sorted(
            model_id
            for model_id, representative_model_id in model_id_mapping.items()
            if model_id != representative_model_id
        )
    )
    model_consolidation = ModelConsolidation(
        model_clusters=model_clusters,
        model_id_mapping=model_id_mapping,
        absorbed_model_ids=absorbed_model_ids,
    )

    model_clustering_record_store.append_clustering_observations(
        pair_observations=pair_clustering_observations,
        model_observations=model_clustering_observations,
    )
    for (
        representative_model_id,
        consolidated_parameters,
    ) in consolidated_parameters_by_representative_model_id.items():
        global_model_repository.set_global_model_parameters(
            model_id=representative_model_id, parameter_snapshot=consolidated_parameters
        )
        if representative_model_id in consolidated_loss_statistics_by_representative_model_id:
            global_model_repository.set_global_model_loss_statistics(
                model_id=representative_model_id,
                loss_statistics=consolidated_loss_statistics_by_representative_model_id[
                    representative_model_id
                ],
            )
    for absorbed_model_id in absorbed_model_ids:
        global_model_repository.remove_global_model(model_id=absorbed_model_id)
    return model_consolidation
