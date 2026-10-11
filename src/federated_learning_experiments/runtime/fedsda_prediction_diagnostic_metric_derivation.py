"""最終構成のFedSDAの参加者の記録から、予測の診断（実際の予測と、比較用の予測の、正否の集計）の指標を導出する。

読取りだけを行い、参加者の状態と乱数を変えない。
"""

from dataclasses import dataclass

from federated_learning_experiments.evaluation.prediction_diagnostic_metric_calculations import (
    ClassMacroPredictionMetrics,
    PredictionCorrectCounts,
    StableAndRecoveryPeriodAccuracies,
    calculate_stable_and_recovery_period_accuracies,
    summarize_class_prediction_metrics,
)
from federated_learning_experiments.evaluation.sample_prediction_record_store import (
    SamplePredictionRecord,
)
from federated_learning_experiments.runtime.fedsda_run_client import FedsdaRunClient


@dataclass(frozen=True, kw_only=True)
class FedsdaPredictionDiagnosticMetrics:
    """予測の診断の指標。数と回数は、全clientの合計。

    「実際の予測」（combined prediction）は、予測に使う重み（設定により、Fixed-Shareの重みか、AdaHedgeの重み）で、
    全保有モデルの出力を結合した予測。Fixed-Shareの重みでの結合と、AdaHedgeの重みでの結合のうち、予測に使う側の指標は、
    実際の予測の指標と同じ値になる。使わない側と、ほかの種類の予測（どれかのモデルか実際の予測、最大重みのモデル、
    確信度が最大のモデル、真の概念別の診断重みでの結合）は、比較のためのもので、予測と学習の判断には使われていない。
    正解の割合・差・比は、数から計算できるので、持たない。
    """

    prediction_correct_counts: PredictionCorrectCounts
    fixed_share_weighted_prediction_correct_count: int
    adahedge_weighted_prediction_correct_count: int
    # 真の概念が分かる標本の数と、そのうち、真の概念別の診断重みでの結合が正解した数。
    true_concept_known_sample_count: int
    true_concept_diagnostic_prediction_correct_count: int
    # 概念の変更の後の回復の区間と、それ以外の定常の区間の、正解の割合。
    combined_prediction_period_accuracies: StableAndRecoveryPeriodAccuracies
    any_model_or_combined_prediction_period_accuracies: StableAndRecoveryPeriodAccuracies
    maximum_weight_model_prediction_period_accuracies: StableAndRecoveryPeriodAccuracies
    fixed_share_weighted_prediction_period_accuracies: StableAndRecoveryPeriodAccuracies
    # 真の概念が分かる標本だけを数える。
    true_concept_diagnostic_prediction_period_accuracies: StableAndRecoveryPeriodAccuracies
    # クラスごとの数（全clientの合計）からの、クラスの間のまとめ。
    class_macro_prediction_metrics: ClassMacroPredictionMetrics
    # 標本ごとの実効モデル数（予測重みの二乗和の逆数）の平均。予測した標本がなければ0。
    mean_effective_model_count: float
    # 同じ平均を、Fixed-Shareの重みで計算したもの。
    mean_fixed_share_effective_model_count: float
    # 予測重みの、最大重みのモデルが変わった回数と、モデルの集まりが変わった回数。
    fixed_share_weight_leader_switch_count: int
    fixed_share_weight_model_pool_reset_count: int
    # AdaHedgeの重みの、集約後の再較正の回数。
    adahedge_weight_aggregation_recalibration_count: int


def _count_prediction_correctness(
    *, prediction_records: tuple[SamplePredictionRecord, ...] | list[SamplePredictionRecord]
) -> PredictionCorrectCounts:
    return PredictionCorrectCounts(
        predicted_sample_count=len(prediction_records),
        combined_prediction_correct_count=sum(
            prediction_record.combined_prediction_is_correct
            for prediction_record in prediction_records
        ),
        any_model_or_combined_prediction_correct_count=sum(
            prediction_record.any_model_or_combined_prediction_is_correct
            for prediction_record in prediction_records
        ),
        maximum_weight_model_prediction_correct_count=sum(
            prediction_record.maximum_weight_model_prediction_is_correct
            for prediction_record in prediction_records
        ),
        highest_confidence_model_prediction_correct_count=sum(
            prediction_record.highest_confidence_model_prediction_is_correct
            for prediction_record in prediction_records
        ),
    )


def derive_fedsda_prediction_diagnostic_metrics(
    *,
    run_clients: tuple[FedsdaRunClient, ...],
    concept_change_sample_indices_by_client: tuple[tuple[int, ...], ...],
    recovery_window_sample_count: int,
) -> FedsdaPredictionDiagnosticMetrics:
    """参加者の、標本ごとの予測の記録と、重みの更新の計数から、予測の診断の指標を作る。

    概念の変更位置は、clientごとの、概念列の全体（処理されなかった末尾を含む）から取った位置。
    """
    if type(run_clients) is not tuple:
        raise TypeError("run_clients must be exact tuple")
    for run_client in run_clients:
        if type(run_client) is not FedsdaRunClient:
            raise TypeError("run_clients must hold exact FedsdaRunClient")
    prediction_records_by_client = tuple(
        run_client.owners.sample_prediction_record_store.snapshot_sample_prediction_records()
        for run_client in run_clients
    )
    all_prediction_records = [
        prediction_record
        for prediction_records in prediction_records_by_client
        for prediction_record in prediction_records
    ]

    def calculate_period_accuracies(
        correctness_field_name: str, *, only_true_concept_known_samples: bool = False
    ) -> StableAndRecoveryPeriodAccuracies:
        # 引数の検査（clientの数、変更位置、窓の件数）は、計算の関数が行う。
        return calculate_stable_and_recovery_period_accuracies(
            prediction_correctness_by_client=tuple(
                tuple(
                    # 真の概念が分からない標本の値（None）は、数えないので、偽にしておく。
                    bool(getattr(prediction_record, correctness_field_name))
                    for prediction_record in prediction_records
                )
                for prediction_records in prediction_records_by_client
            ),
            concept_change_sample_indices_by_client=concept_change_sample_indices_by_client,
            recovery_window_sample_count=recovery_window_sample_count,
            sample_inclusion_by_client=(
                tuple(
                    tuple(
                        prediction_record.observed_concept_id is not None
                        for prediction_record in prediction_records
                    )
                    for prediction_records in prediction_records_by_client
                )
                if only_true_concept_known_samples
                else None
            ),
        )

    # クラスごとの記録（クラスは、clientの順・標本の順に、最初に現れた順）。
    prediction_records_by_class_id: dict[int, list[SamplePredictionRecord]] = {}
    for prediction_record in all_prediction_records:
        prediction_records_by_class_id.setdefault(prediction_record.observed_class_id, []).append(
            prediction_record
        )

    # 実効モデル数の平均（clientごとに、標本の順に足し、clientの順に足す）。
    def calculate_mean_effective_model_count(effective_model_count_field_name: str) -> float:
        effective_model_count_sum = 0.0
        for prediction_records in prediction_records_by_client:
            client_effective_model_count_sum = 0.0
            for prediction_record in prediction_records:
                client_effective_model_count_sum += getattr(
                    prediction_record, effective_model_count_field_name
                )
            effective_model_count_sum += client_effective_model_count_sum
        return (
            effective_model_count_sum / len(all_prediction_records)
            if all_prediction_records
            else 0.0
        )

    return FedsdaPredictionDiagnosticMetrics(
        prediction_correct_counts=_count_prediction_correctness(
            prediction_records=all_prediction_records
        ),
        fixed_share_weighted_prediction_correct_count=sum(
            prediction_record.fixed_share_weighted_prediction_is_correct
            for prediction_record in all_prediction_records
        ),
        adahedge_weighted_prediction_correct_count=sum(
            prediction_record.adahedge_weighted_prediction_is_correct
            for prediction_record in all_prediction_records
        ),
        true_concept_known_sample_count=sum(
            prediction_record.observed_concept_id is not None
            for prediction_record in all_prediction_records
        ),
        true_concept_diagnostic_prediction_correct_count=sum(
            bool(prediction_record.true_concept_diagnostic_prediction_is_correct)
            for prediction_record in all_prediction_records
        ),
        combined_prediction_period_accuracies=calculate_period_accuracies(
            "combined_prediction_is_correct"
        ),
        any_model_or_combined_prediction_period_accuracies=calculate_period_accuracies(
            "any_model_or_combined_prediction_is_correct"
        ),
        maximum_weight_model_prediction_period_accuracies=calculate_period_accuracies(
            "maximum_weight_model_prediction_is_correct"
        ),
        fixed_share_weighted_prediction_period_accuracies=calculate_period_accuracies(
            "fixed_share_weighted_prediction_is_correct"
        ),
        true_concept_diagnostic_prediction_period_accuracies=calculate_period_accuracies(
            "true_concept_diagnostic_prediction_is_correct", only_true_concept_known_samples=True
        ),
        class_macro_prediction_metrics=summarize_class_prediction_metrics(
            class_prediction_correct_counts=tuple(
                _count_prediction_correctness(prediction_records=class_prediction_records)
                for class_prediction_records in prediction_records_by_class_id.values()
            )
        ),
        mean_effective_model_count=calculate_mean_effective_model_count("effective_model_count"),
        mean_fixed_share_effective_model_count=calculate_mean_effective_model_count(
            "fixed_share_effective_model_count"
        ),
        fixed_share_weight_leader_switch_count=sum(
            run_client.owners.fixed_share_weight_controller.leader_switch_count
            for run_client in run_clients
        ),
        fixed_share_weight_model_pool_reset_count=sum(
            run_client.owners.fixed_share_weight_controller.model_pool_reset_count
            for run_client in run_clients
        ),
        adahedge_weight_aggregation_recalibration_count=sum(
            run_client.owners.adahedge_weight_evidence_collection.all_sample_weight_evidence.aggregation_recalibration_count
            for run_client in run_clients
        ),
    )
